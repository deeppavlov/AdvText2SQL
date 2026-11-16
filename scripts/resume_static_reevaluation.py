"""Connection-only supervisor for frozen, LLM-free BIRD reevaluation.

Does not import checkpoints under a new SHA or modify the scorer. Existing
external tunnels are checked but never killed. Only confirmed loss of database
access (or death of our own SSH process) permits a bounded worker restart.
"""

import argparse
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time

from dotenv import load_dotenv

import static_pipeline_regression as audit


class AccessFailure(RuntimeError):
    def __init__(self, reason, *, retryable, sqlstate=None):
        super().__init__(reason)
        self.reason, self.retryable, self.sqlstate = reason, retryable, sqlstate


def port_open():
    with socket.socket() as connection:
        connection.settimeout(0.5)
        return connection.connect_ex(("127.0.0.1", 5444)) == 0


def probe_database(database):
    from sqlalchemy import text
    from sqlalchemy.engine import URL
    from baseline_connection import database_engine
    from adv_text2sql.evaluation import database_failure
    from adv_text2sql.sql_safety import read_only_connection
    from resilient_hybrid_runtime import connection_reason

    engine = database_engine(
        URL.create(
            "postgresql+psycopg",
            username=os.environ["DB_USER"],
            password=os.environ["DB_PASS"],
            host="127.0.0.1",
            port=5444,
            database=database,
        )
    )
    try:
        with read_only_connection(engine, "SELECT 1") as connection:
            if connection.execute(text("SELECT 1")).scalar() != 1:
                raise AccessFailure("unexpected_readiness_result", retryable=False)
    except AccessFailure:
        raise
    except Exception as error:
        details = database_failure(error)
        reason, _ = connection_reason(error)
        code = details["sqlstate"]
        retry = (
            isinstance(code, str)
            and code.startswith("08")
            or code is None
            and reason
            in {
                "tunnel_port_unavailable",
                "connection_closed",
                "connection_or_statement_timeout",
            }
        )
        # No retries for authentication, permissions, missing DBs or SQLSTATE
        # 57014 statement cancellation, even if its message mentions timeout.
        raise AccessFailure(reason, retryable=bool(retry), sqlstate=code) from None
    finally:
        engine.dispose()


class Tunnel:
    def __init__(self, log):
        self.log, self.process, self.borrowed = log, None, False

    def start(self):
        if port_open():
            self.borrowed = True
            print(
                "[CONNECTION] Checking existing port 5444; external process will not be stopped.",
                flush=True,
            )
            return
        git_ssh = Path("C:/Program Files/Git/usr/bin/ssh.exe")
        executable = (
            str(git_ssh)
            if os.name == "nt" and git_ssh.is_file()
            else shutil.which("ssh")
        )
        if not executable:
            raise AccessFailure("ssh_not_installed", retryable=False)
        command = [
            executable,
            "-N",
            "-o",
            "BatchMode=yes",
            "-o",
            "ExitOnForwardFailure=yes",
            "-o",
            "ConnectTimeout=10",
            "-o",
            "ConnectionAttempts=1",
            "-o",
            "ServerAliveInterval=10",
            "-o",
            "ServerAliveCountMax=3",
            "-L",
            "127.0.0.1:5444:10.11.1.6:5444",
            "-p",
            "2278",
            "onik110@lnsigo.mipt.ru",
        ]
        self.process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=self.log,
            stderr=self.log,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)
            if os.name == "nt"
            else 0,
        )
        for _ in range(40):
            if self.process.poll() is not None:
                raise AccessFailure("managed_ssh_stopped", retryable=True)
            if port_open():
                return
            time.sleep(0.5)
        raise AccessFailure("managed_ssh_not_ready", retryable=True)

    def died(self):
        return self.process is not None and self.process.poll() is not None

    def close(self):
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)


def worker_args(root):
    return [
        sys.executable,
        "-u",
        str(audit.ROOT / "scripts/static_pipeline_regression.py"),
        "--action",
        "reevaluate",
        "--benchmark",
        "bird",
        "--output",
        str(root),
    ]


def supervise(root, manifest, *, max_restarts=2):
    databases = sorted(
        {item["db_id"] for item in manifest["benchmarks"]["bird"]["items"]}
    )
    records = audit.rescore_records(root, manifest, "bird")
    if len(records) == len(manifest["benchmarks"]["bird"]["items"]):
        if any(row["status"] != "scored" for row in records.values()):
            raise AccessFailure("reference_errors_require_review", retryable=False)
        print("[COMPLETE] All BIRD answers already saved; no SSH, DB or LLM calls.")
        return 0
    state = {
        "manifest_sha256": manifest["sha256"],
        "llm_calls": 0,
        "saved_before": len(records),
        "supervisor_sha256": audit.digest(Path(__file__)),
        "max_restarts": max_restarts,
        "attempts": [],
    }
    journal = root / "connection_resume.json"
    with (root / "connection_resume_tunnel.log").open("a", encoding="utf-8") as log:
        for attempt in range(max_restarts + 1):
            tunnel = Tunnel(log)
            entry = {"attempt": attempt + 1, "at_utc": audit.source.search.now()}
            state["attempts"].append(entry)
            failure = None
            try:
                audit.validate(manifest, model=True)
                tunnel.start()
                for index, database in enumerate(databases, 1):
                    print(
                        f"[DB READY {index}/{len(databases)}] {database}: SELECT 1",
                        flush=True,
                    )
                    probe_database(database)
                print(
                    "[RESUME] Database access verified; running the original frozen evaluator.",
                    flush=True,
                )
                code = subprocess.run(
                    worker_args(root), cwd=audit.ROOT, check=False
                ).returncode
                entry["worker_exit_code"] = code
                if code == 0:
                    state["status"] = "completed"
                    return 0
                if code in (130, -2, 0xC000013A, -1073741510):
                    state["status"] = "interrupted"
                    return code
                if tunnel.died():
                    failure = AccessFailure("managed_ssh_stopped", retryable=True)
                else:
                    # If SELECT 1 succeeds, the error may be a query timeout or
                    # a schema/SQL/programming issue. Do not blindly retry it.
                    progress_path = root / "bird/reevaluation_progress.json"
                    progress = (
                        audit.read(progress_path) if progress_path.exists() else {}
                    )
                    database = progress.get("current_database", databases[0])
                    if database not in databases:
                        raise AccessFailure(
                            "foreign_progress_database", retryable=False
                        )
                    try:
                        probe_database(database)
                    except AccessFailure as error:
                        failure = error
                    else:
                        print(
                            "[STOP] Worker failed while DB access is healthy; no automatic query retry.",
                            flush=True,
                        )
                        state["status"] = "worker_error_with_healthy_connection"
                        return code
            except AccessFailure as error:
                failure = error
            finally:
                tunnel.close()
                if failure is not None:
                    entry.update(
                        reason=failure.reason,
                        sqlstate=failure.sqlstate,
                        retryable=failure.retryable,
                        external_tunnel=tunnel.borrowed,
                    )
                audit.store.write_json(journal, state)
            if tunnel.borrowed:
                print(
                    "[STOP] Existing external tunnel is unavailable; stop its owner and repeat this command.",
                    flush=True,
                )
                state["status"] = "external_tunnel_unavailable"
                audit.store.write_json(journal, state)
                return 1
            if not failure.retryable or attempt == max_restarts:
                print(
                    f"[STOP] Database access unavailable: {failure.reason}; checkpoint preserved.",
                    flush=True,
                )
                state["status"] = "connection_unavailable"
                audit.store.write_json(journal, state)
                return 1
            print(
                f"[RECONNECT] {failure.reason}; restarting only our SSH tunnel in 3s "
                f"({attempt + 1}/{max_restarts}); no LLM calls.",
                flush=True,
            )
            time.sleep(3)
    return 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", default="research_results/static_pipeline_audit_numeric_20261116"
    )
    parser.add_argument("--max-restarts", type=int, choices=range(6), default=2)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    root = audit.output_path(args.output)
    manifest = audit.read(root / "experiment.json")
    audit.validate(manifest, model=True)
    records = audit.rescore_records(root, manifest, "bird")
    total = len(manifest["benchmarks"]["bird"]["items"])
    print(
        f"BIRD saved={len(records)}/{total}, remaining={total - len(records)}; LLM calls=0",
        flush=True,
    )
    if args.dry_run:
        print("OFFLINE: protocol/checkpoint validated; no SSH or DB calls.")
        return 0
    load_dotenv(audit.ROOT / manifest["env_file"], override=True)
    supervisor_lock = root / "connection_supervisor"
    supervisor_lock.mkdir(exist_ok=True)
    with audit.suite_lock(supervisor_lock):
        with audit.suite_lock(root):
            # Check for an existing worker, but let our child take its own
            # checkpoint lock. The supervisor has a separate lifetime lock.
            pass
        return supervise(root, manifest, max_restarts=args.max_restarts)


if __name__ == "__main__":
    sys.exit(main())
