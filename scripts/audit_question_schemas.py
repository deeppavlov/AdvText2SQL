"""Offline reference-to-schema checks and output re-scoring; no SQL or LLM execution."""

import argparse
import ast
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import sqlglot
from sqlglot import exp
from sqlglot.optimizer.scope import traverse_scope
from sqlglot.errors import ParseError

from atomic_artifacts import write_json
from build_configuration_dataset import tsv_schemas
from sql_output_contract import VERSION, sql_output_issue
import static_hybrid_search as search

ROOT = search.ROOT
DEFAULT_OUTPUT = (
    "research_results/deepseek_combinations_large_20261026/quality_audit_20261029"
)


def schema_snapshots(items, raw_path):
    candidates = defaultdict(set)
    by_question = defaultdict(set)
    for i in items:
        by_question[i["question"]].add(i["db_id"])
    for line in raw_path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        blocks = set(tsv_schemas(row["messages"]))
        if not blocks:
            continue
        explicit = row.get("db_id")
        dbs = (
            {explicit}
            if explicit
            else {
                db
                for question, targets in by_question.items()
                if any(question in m["content"] for m in row["messages"])
                for db in targets
            }
        )
        if len(dbs) == 1:
            candidates[next(iter(dbs))].update(blocks)
    result = {}
    for db in sorted({i["db_id"] for i in items}):
        if len(candidates[db]) != 1:
            raise ValueError(f"Missing/inconsistent historical schema: {db}")
        rows = list(next(iter(candidates[db])))
        result[db] = {"rows": rows, "schema_sha256": search.fingerprint(rows)}
    return result


def sql_references(item):
    if item["SQL"] != "ambiguous":
        return [item["SQL"]], None
    value = item.get("ambig_queries")
    if isinstance(value, str):
        for parser in (json.loads, ast.literal_eval):
            try:
                value = parser(value)
                break
            except (ValueError, SyntaxError):
                continue
    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(v, str) for v in value)
    ):
        return [], "missing_or_invalid_reference_alternatives"
    # Some saved alternatives contain literal backslash-r/backslash-n escapes.
    return [v.replace("\\r\\n", "\n").replace("\\n", "\n") for v in value], None


def reference_issues(sql, rows):
    tables = defaultdict(set)
    for table, column, _ in rows:
        tables[table.casefold()].add(column.casefold())
    try:
        # Both benchmark reference archives originate from SQLite (backticks/DQS).
        statements = sqlglot.parse(sql, read="sqlite")
    except ParseError:
        return {"status": "review", "issues": ["reference_parse_error"]}
    if len(statements) != 1 or not isinstance(statements[0], exp.Query):
        return {"status": "review", "issues": ["reference_not_single_query"]}
    missing_tables, missing_columns = set(), set()
    unresolved, notes = set(), set()
    for scope in traverse_scope(statements[0]):
        physical = {
            alias.casefold(): t.name.casefold()
            for alias, t in scope.sources.items()
            if isinstance(t, exp.Table)
        }
        derived = {
            alias.casefold()
            for alias, t in scope.sources.items()
            if not isinstance(t, exp.Table)
        }
        for table in physical.values():
            if table not in tables:
                missing_tables.add(table)
        projection_aliases = {
            v.alias.casefold() for v in scope.expression.selects if v.alias
        }
        for col in scope.columns:
            # SQLGlot propagates unqualified child columns to an outer scope until
            # qualification. Audit them where they occur, never against outer tables.
            if col.find_ancestor(exp.Query) is not scope.expression:
                continue
            name, alias = col.name.casefold(), col.table.casefold()
            if name == "*":
                continue
            if alias in physical:
                table = physical[alias]
                if table in tables and name not in tables[table]:
                    missing_columns.add(f"{table}.{name}")
            elif alias and alias not in derived:
                # Correlated references may bind to a parent; never call those a missing column.
                parent = scope.parent
                while parent and alias not in {a.casefold() for a in parent.sources}:
                    parent = parent.parent
                if parent is None:
                    unresolved.add(alias)
            elif (
                not alias
                and not derived
                and physical
                and name not in projection_aliases
            ):
                present = [t for t in physical.values() if t in tables]
                if (
                    present
                    and not missing_tables
                    and not any(name in tables[t] for t in present)
                ):
                    # SQLite permits double-quoted string literals if no matching
                    # identifier exists. Restrict that fallback to value positions.
                    parent = col.parent
                    dqs_value = col.this.args.get("quoted") and (
                        (isinstance(parent, exp.Binary) and parent.expression is col)
                        or (isinstance(parent, exp.In) and col in parent.expressions)
                        or (isinstance(parent, exp.Between) and col is not parent.this)
                    )
                    if dqs_value:
                        notes.add("sqlite_double_quoted_literal")
                    else:
                        missing_columns.add(name)
    issues = (
        ["missing_tables:" + ",".join(sorted(missing_tables))] if missing_tables else []
    )
    issues += (
        ["missing_columns:" + ",".join(sorted(missing_columns))]
        if missing_columns
        else []
    )
    issues += (
        ["unresolved_aliases:" + ",".join(sorted(unresolved))] if unresolved else []
    )
    return {
        "status": "mismatch"
        if missing_tables or missing_columns
        else "review"
        if unresolved
        else "reference_consistent",
        "issues": issues,
        "notes": sorted(notes),
    }


def audit_items(items, snapshots):
    result = []
    for item in items:
        refs, issue = sql_references(item)
        analyses = [
            reference_issues(sql, snapshots[item["db_id"]]["rows"]) for sql in refs
        ]
        status = (
            "review"
            if issue
            else "mismatch"
            if any(a["status"] == "mismatch" for a in analyses)
            else "review"
            if any(a["status"] == "review" for a in analyses)
            else "reference_consistent"
        )
        result.append(
            {
                "question_id": str(item["question_id"]),
                "db_id": item["db_id"],
                "status": status,
                "reference_issue": issue,
                "reference_checks": analyses,
            }
        )
    return result


def audit(output, source_output):
    manifest = search.read(source_output / "experiment.json")
    search.validate_manifest(manifest)
    snapshots, audits, sources = (
        {},
        {},
        [
            source_output / "experiment.json",
            Path(__file__).resolve(),
            ROOT / "scripts/sql_output_contract.py",
        ],
    )
    for b, dataset in search.DATA_FILES.items():
        items = search.read(ROOT / dataset)
        # The taxonomy template contains benchmark-like examples; use the canonical
        # FEAT_35 archive (also used by the original dataset builder) for DB identity.
        path = (
            ROOT
            / "research_results/deepseek_direct_ablation_large_20261024/feat_35"
            / ("bird" if b == "bird" else "ambrosia_227")
            / "raw_responses.jsonl"
        )
        snapshots[b] = schema_snapshots(items, path)
        audits[b] = audit_items(items, snapshots[b])
        sources += [ROOT / dataset, path]
    corrected = []
    invalid = []
    for case in manifest["cases"]:
        values = {}
        for b in search.BENCHMARKS:
            folder = search.run_folder(source_output, "train", case["name"], b)
            if (
                not (folder / "metadata.json").exists()
                or search.read(folder / "metadata.json")["status"] != "completed"
            ):
                break
            legacy = search.result(source_output, manifest, "train", case, b)
            predictions = search.read(folder / "query_results.json")
            bad = {
                q: sql_output_issue(v)
                for q, v in predictions.items()
                if v not in ("error", "ambiguous") and sql_output_issue(v)
            }
            for q, reason in bad.items():
                invalid.append(
                    {
                        "case": case["name"],
                        "benchmark": b,
                        "question_id": q,
                        "reason": reason,
                        "response": predictions[q],
                    }
                )
            normalized = {q: "error" if q in bad else v for q, v in predictions.items()}
            if b == "ambrosia":
                from research_metrics import ambrosia_report

                fixed = ambrosia_report(manifest["cohorts"]["train"][b], normalized)
                score, f1 = fixed["accuracy"], fixed["ambiguity_metrics"]["f1"]
            else:
                score = (
                    100
                    * sum(v for q, v in legacy["outcomes"].items() if q not in bad)
                    / legacy["questions"]
                )
                f1 = None
            allowed = {
                q["question_id"]
                for q in audits[b]
                if q["status"] == "reference_consistent"
            }
            subset = [
                i
                for i in manifest["cohorts"]["train"][b]
                if str(i["question_id"]) in allowed
            ]
            clean_predictions = {q: v for q, v in normalized.items() if q in allowed}
            if b == "ambrosia":
                clean_score = ambrosia_report(subset, clean_predictions)["accuracy"]
            else:
                clean_score = (
                    100
                    * sum(
                        v
                        for q, v in legacy["outcomes"].items()
                        if q in allowed and q not in bad
                    )
                    / len(subset)
                )
            values[b] = {
                "original_accuracy": legacy["accuracy"],
                "contract_accuracy": score,
                "invalid_answers": len(bad),
                "f1": f1,
                "questions": legacy["questions"],
                "reference_consistent_questions": len(subset),
                "reference_consistent_accuracy": clean_score,
            }
            sources += [
                folder / f
                for f in ("metadata.json", "query_results.json", "report.json")
            ]
        if len(values) == 2:
            corrected.append(
                {
                    "case": case["name"],
                    "features": case["features"],
                    "benchmarks": values,
                    "macro_accuracy": sum(
                        v["contract_accuracy"] for v in values.values()
                    )
                    / 2,
                }
            )
    sources = list(dict.fromkeys(sources))
    report = {
        "format": "schema-and-output-audit-v1",
        "output_contract": VERSION,
        "scope": "all frozen dataset questions for schema audit; train for result re-scoring",
        "schemas": snapshots,
        "questions": audits,
        "invalid_answers": invalid,
        "contract_rescored_results": sorted(
            corrected, key=lambda r: -r["macro_accuracy"]
        ),
        "source_sha256": {
            p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sources
        },
        "limitations": [
            "Static comparison to saved schemas, not a live catalog or semantic equivalence proof.",
            "Gold SQL is used only in this offline audit, never in model prompts.",
            "No missing tables/types/data reconstructed from gold SQL; original database archives are needed for restoration.",
            "Historical outputs/protocols are preserved; re-scored metrics use a different declared output contract.",
        ],
    }
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "audit.json", report)
    write_json(output / "schema_snapshot.json", snapshots)
    lines = [
        "# Schema and SQL-output audit",
        "",
        "Offline; no API or SQL execution. Original results remain unchanged.",
        "",
        "| Benchmark | Questions | Reference-consistent | Mismatch | Review |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for b, questions in audits.items():
        lines.append(
            f"| {b} | {len(questions)} | {sum(q['status'] == 'reference_consistent' for q in questions)} | "
            f"{sum(q['status'] == 'mismatch' for q in questions)} | {sum(q['status'] == 'review' for q in questions)} |"
        )
    lines += [
        "",
        "## Inconsistent databases",
        "",
        "| Benchmark | Database | Affected questions | Issues |",
        "| --- | --- | ---: | --- |",
    ]
    for b, questions in audits.items():
        for db in sorted(
            {q["db_id"] for q in questions if q["status"] != "reference_consistent"}
        ):
            affected = [
                q
                for q in questions
                if q["db_id"] == db and q["status"] != "reference_consistent"
            ]
            issues = sorted(
                {
                    i
                    for q in affected
                    for ref in q["reference_checks"]
                    for i in ref["issues"]
                }
            )
            lines.append(f"| {b} | {db} | {len(affected)} | {'; '.join(issues)} |")
    lines += [
        "",
        "## Output-contract re-scoring (same train IDs, mismatches not excluded)",
        "",
        "| Case | BIRD original / contract | Ambrosia original / contract | Invalid BIRD / Ambrosia | Contract mean |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for r in report["contract_rescored_results"]:
        b, a = r["benchmarks"]["bird"], r["benchmarks"]["ambrosia"]
        lines.append(
            f"| {r['case']} | {b['original_accuracy']:.2f} / {b['contract_accuracy']:.2f} | "
            f"{a['original_accuracy']:.2f} / {a['contract_accuracy']:.2f} | {b['invalid_answers']} / {a['invalid_answers']} | {r['macro_accuracy']:.2f} |"
        )
    lines += [
        "",
        "## Separate reference-consistent train subset (offline replay, not new runs)",
        "",
        "| Case | BIRD | Ambrosia | Mean |",
        "| --- | ---: | ---: | ---: |",
    ]
    for r in sorted(
        report["contract_rescored_results"],
        key=lambda r: -sum(
            v["reference_consistent_accuracy"] for v in r["benchmarks"].values()
        ),
    ):
        b, a = r["benchmarks"]["bird"], r["benchmarks"]["ambrosia"]
        lines.append(
            f"| {r['case']} | {b['reference_consistent_accuracy']:.2f} ({b['reference_consistent_questions']}) | "
            f"{a['reference_consistent_accuracy']:.2f} ({a['reference_consistent_questions']}) | "
            f"{(b['reference_consistent_accuracy'] + a['reference_consistent_accuracy']) / 2:.2f} |"
        )
    lines += [
        "",
        "Passing output shape does not prove SQL correctness. Syntax/execution remains the job of FEAT_16/evaluation.",
        "A schema mismatch is a dataset/setup problem, not evidence that a feature is harmful.",
        "",
    ]
    (output / "comparison.md").write_text(
        "\n".join(lines), encoding="utf-8", newline="\n"
    )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--search", default=search.DEFAULT_OUTPUT)
    parser.add_argument("--out", default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output, source_output = (ROOT / args.out).resolve(), (ROOT / args.search).resolve()
    if (
        not output.is_relative_to(ROOT / "research_results")
        or output == ROOT / "research_results"
        or output == source_output
        or output in source_output.parents
    ):
        parser.error(
            "Use a dedicated audit folder inside research_results, separate from the source run"
        )
    report = audit(output, source_output)
    print(
        {
            b: {
                status: sum(q["status"] == status for q in qs)
                for status in ("reference_consistent", "mismatch", "review")
            }
            for b, qs in report["questions"].items()
        }
    )
    print("Invalid saved answers:", len(report["invalid_answers"]))
