"""Deterministic database/family partitions; never use configuration scores."""

from collections import Counter, defaultdict
from difflib import SequenceMatcher
import hashlib
import random
import re
import unicodedata

PARTITIONS = ("train", "dev", "test")
FAMILY_POLICY = "nfkc-casefold-quoted-literals-numbers-v1"


def digest(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:20]


def normalize_family(text, replace_literals=True):
    text = unicodedata.normalize("NFKC", text).casefold()
    if replace_literals:
        text = re.sub(r'"[^"\n]*"|(?<!\w)\x27[^\x27\n]*\x27(?!\w)', " literal ", text)
    text = re.sub(r"\b\d+(?:[.,:/-]\d+)*\b", " number ", text)
    return " ".join(re.findall(r"\w+", text))


def family_keys(item, benchmark):
    keys = {"text:" + digest(normalize_family(item["question"]))}
    if benchmark == "ambrosia" and item.get("ambig_question"):
        keys.add("ambrosia:" + digest(normalize_family(item["ambig_question"])))
    return sorted(keys)


def make_groups(records):
    """Union same DB, normalized question/family, and conservative near-duplicates."""
    parent = {r["id"]: r["id"] for r in records}

    def find(key):
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    def union(first, second):
        a, b = find(first), find(second)
        if a != b:
            parent[max(a, b)] = min(a, b)

    seen = {}
    for row in sorted(records, key=lambda r: r["id"]):
        keys = ["db:" + row["benchmark"] + ":" + row["db_id"], *row["family_keys"]]
        for key in keys:
            if key in seen:
                union(row["id"], seen[key])
            else:
                seen[key] = row["id"]
    normalized = {r["id"]: normalize_family(r["inputs"]["question"]) for r in records}
    tokens = {key: set(text.split()) for key, text in normalized.items()}
    near_edges = []
    ordered = sorted(records, key=lambda r: r["id"])
    for i, a in enumerate(ordered):
        for b in ordered[i + 1:]:
            if (a["benchmark"], a["db_id"]) == (b["benchmark"], b["db_id"]):
                continue
            x, y = tokens[a["id"]], tokens[b["id"]]
            if min(len(x), len(y)) < 8 or len(x & y) / len(x | y) < 0.8:
                continue
            ratio = SequenceMatcher(None, normalized[a["id"]], normalized[b["id"]], autojunk=False).ratio()
            if ratio >= 0.9:
                union(a["id"], b["id"])
                near_edges.append([a["id"], b["id"]])
    components = defaultdict(list)
    for row in records:
        components[find(row["id"])].append(row["id"])
    groups = {}
    assignments = {}
    for ids in components.values():
        ids = sorted(ids)
        group = "group:" + digest("\n".join(ids))
        groups[group] = ids
        assignments.update({key: group for key in ids})
    return assignments, dict(sorted(groups.items())), near_edges


def partition(records, groups, seed=42, ratios=(0.6, 0.2, 0.2), attempts=128):
    if len(ratios) != 3 or any(v <= 0 for v in ratios) or abs(sum(ratios) - 1) > 1e-8:
        raise ValueError("Three positive split ratios must sum to 1")
    by_id = {r["id"]: r for r in records}
    benchmarks = sorted({r["benchmark"] for r in records})
    totals = Counter(r["benchmark"] for r in records)
    weights = {g: Counter(by_id[i]["benchmark"] for i in ids) for g, ids in groups.items()}
    if any(sum(bool(w[b]) for w in weights.values()) < 3 for b in benchmarks):
        raise ValueError("Need at least three independent database/family groups per benchmark")
    targets = {s: {b: totals[b] * ratios[j] for b in benchmarks} for j, s in enumerate(PARTITIONS)}

    def score(counts, require_nonempty=True):
        value = sum(((counts[s][b] - targets[s][b]) / totals[b]) ** 2
                    for s in PARTITIONS for b in benchmarks)
        if require_nonempty:
            value += 10 * sum(counts[s][b] == 0 for s in PARTITIONS for b in benchmarks)
        return value

    rng = random.Random(seed)
    best = None
    for _ in range(attempts):
        order = sorted(groups)
        rng.shuffle(order)
        order.sort(key=lambda g: max(weights[g][b] / totals[b] for b in benchmarks), reverse=True)
        counts = {s: Counter() for s in PARTITIONS}
        selected = {}
        for group in order:
            choices = list(PARTITIONS)
            rng.shuffle(choices)
            def after(split):
                counts[split].update(weights[group])
                value = score(counts, require_nonempty=False)
                counts[split].subtract(weights[group])
                return value
            selected[group] = min(choices, key=after)
            counts[selected[group]].update(weights[group])
        # Local refinement balances counts only, not outcomes or utility labels.
        for _ in range(12):
            improved = False
            for group in order:
                old = selected[group]
                initial = score(counts)
                for new in PARTITIONS:
                    if new == old:
                        continue
                    counts[old].subtract(weights[group])
                    counts[new].update(weights[group])
                    if score(counts) < initial - 1e-12:
                        selected[group] = new
                        improved = True
                        break
                    counts[new].subtract(weights[group])
                    counts[old].update(weights[group])
            if not improved:
                break
        signature = tuple(sorted(selected.items()))
        candidate = (score(counts), signature)
        if best is None or candidate < best:
            best = candidate
    assignments = dict(best[1])
    splits = {s: sorted(i for g, ids in groups.items() if assignments[g] == s for i in ids) for s in PARTITIONS}
    audit_partitions(records, groups, splits)
    return splits


def audit_partitions(records, groups, splits):
    by_id = {r["id"]: r for r in records}
    if len(by_id) != len(records):
        raise ValueError("Record IDs are not unique")
    all_ids = [i for ids in splits.values() for i in ids]
    if len(all_ids) != len(set(all_ids)) or set(all_ids) != set(by_id):
        raise ValueError("Partitions must cover each record exactly once")
    owner = {i: s for s, ids in splits.items() for i in ids}
    if any(len({owner[i] for i in ids}) != 1 for ids in groups.values()):
        raise ValueError("Database/family component crosses partitions")
    db_splits = defaultdict(set)
    family_splits = defaultdict(set)
    for row in records:
        db_splits[(row["benchmark"], row["db_id"])].add(owner[row["id"]])
        for family in row["family_keys"]:
            family_splits[family].add(owner[row["id"]])
    if any(len(v) > 1 for v in db_splits.values()) or any(len(v) > 1 for v in family_splits.values()):
        raise ValueError("Database or question family leakage")
    for benchmark in {r["benchmark"] for r in records}:
        if any(not any(by_id[i]["benchmark"] == benchmark for i in ids) for ids in splits.values()):
            raise ValueError("Every partition must contain both benchmarks")
    return {"id_overlap": 0, "database_overlap": 0, "family_overlap": 0}


def training_folds(records, groups, splits, count=4):
    train = set(splits["train"])
    by_id = {r["id"]: r for r in records}
    selected = {g: ids for g, ids in groups.items() if set(ids) <= train}
    counts = [Counter() for _ in range(count)]
    folds = [[] for _ in range(count)]
    totals = Counter(by_id[i]["benchmark"] for i in train)
    for group, ids in sorted(selected.items(), key=lambda x: (-len(x[1]), x[0])):
        weights = Counter(by_id[i]["benchmark"] for i in ids)
        target = min(range(count), key=lambda f: (
            sum(counts[f][b] / totals[b] for b in weights), len(folds[f]), f))
        folds[target].extend(ids)
        counts[target].update(weights)
    if any(not ids for ids in folds):
        raise ValueError("Not enough training groups for the requested CV folds")
    return [{"fold": i, "train_ids": sorted(train - set(ids)), "validation_ids": sorted(ids)}
            for i, ids in enumerate(folds)]


def family_stress_folds(records):
    """Ambrosia construction-family holdout, separate from the main split."""
    rows = [r for r in records if r["benchmark"] == "ambrosia"]
    families = sorted({r["analysis"]["construction_family"] for r in rows})
    return [{"held_out_family": family,
             "train_ids": sorted(r["id"] for r in rows if r["analysis"]["construction_family"] != family),
             "test_ids": sorted(r["id"] for r in rows if r["analysis"]["construction_family"] == family)}
            for family in families]
