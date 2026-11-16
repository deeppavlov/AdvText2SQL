"""Versioned benchmark checkpoints; a flat prediction file alone is not provenance."""

import hashlib
import json
import os
from pathlib import Path
import tempfile


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=f".{path.name}.", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def identity(benchmark):
    from adv_text2sql.mcp_servers.text2sql_tool.src import text2sql_implementation as impl

    root = Path(__file__).resolve().parents[1]
    model = {k: os.environ.get(k) for k in ("LLM_MODEL_NAME", "LLM_MODEL_VERSION", "LLM_BASE_URL")}
    client, visited = benchmark.llm_client, set()
    while client is not None and id(client) not in visited:
        visited.add(id(client))
        state = vars(client) if hasattr(client, "__dict__") else {}
        config = state.get("_raw_config", {})
        if isinstance(config, dict):
            model.update({k: config[k] for k in ("model", "base_url", "temperature", "max_tokens", "top_p", "seed",
                                                "reasoning_effort", "reasoning", "extra_body", "response_format",
                                                "stop", "frequency_penalty", "presence_penalty")
                          if k in config})
        client = state.get("client", state.get("llm_client"))
    files = [*sorted(Path(__file__).parent.glob("*.py")),
             *sorted((root / "src/adv_text2sql/mcp_servers/text2sql_tool/src").glob("*.py")),
             *sorted((root / "src/adv_text2sql").glob("*.py"))]
    return {
        "benchmark": type(benchmark).__name__,
        "questions_sha256": hashlib.sha256(Path(benchmark.query_file).read_bytes()).hexdigest(),
        "answers_sha256": hashlib.sha256(Path(benchmark.answer_file).read_bytes()).hexdigest(),
        "model_sha256": fingerprint(model),
        "database_endpoint_sha256": fingerprint(benchmark.db_url),
        "feature_flags": {name: value for name, value in vars(impl).items() if name.startswith("FEAT_")},
        "prompts_sha256": fingerprint({name: value for name, value in vars(impl).items()
                                       if "PROMPT" in name and isinstance(value, str)}),
        "code_sha256": fingerprint({str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                                    for p in files}),
        "use_evidence": getattr(benchmark, "use_evidence", False),
        "context_sha256": fingerprint(benchmark.checkpoint_context),
    }


class BenchmarkCheckpoint:
    def __init__(self, benchmark, folder=Path(".")):
        self.folder = Path(folder)
        self.envelope = self.folder / "query_checkpoint.json"
        self.export = self.folder / "query_results.json"
        self.identity = identity(benchmark)
        items = json.loads(Path(benchmark.query_file).read_text(encoding="utf-8"))
        self.ids = {str(item["question_id"]) for item in items}
        if len(self.ids) != len(items):
            raise ValueError("Duplicate question IDs in checkpoint dataset")

    def validate_predictions(self, predictions):
        if not isinstance(predictions, dict) or not set(predictions) <= self.ids:
            raise ValueError("Checkpoint contains foreign question IDs")
        if any(not isinstance(v, str) or not v.strip() for v in predictions.values()):
            raise ValueError("Checkpoint contains invalid predictions")

    def load(self):
        if not self.envelope.exists():
            if self.export.exists():
                raise ValueError("Legacy checkpoint has no verified identity; use a new output directory")
            return {}
        value = json.loads(self.envelope.read_text(encoding="utf-8"))
        if (value.get("version") != "benchmark-checkpoint-v1" or value.get("identity") != self.identity
                or value.get("identity_sha256") != fingerprint(self.identity)):
            raise ValueError("Checkpoint dataset/model/prompts/features/code differ; use a new output directory")
        predictions = value["predictions"]
        self.validate_predictions(predictions)
        if value.get("predictions_sha256") != fingerprint(predictions):
            raise ValueError("Checkpoint prediction hash changed")
        # The envelope is authoritative and written first. A stopped export must
        # be repaired explicitly; the paid responses remain in the envelope.
        if not self.export.exists() or json.loads(self.export.read_text(encoding="utf-8")) != predictions:
            raise ValueError("Checkpoint export differs; restore query_results.json from query_checkpoint.json predictions")
        return predictions

    def save(self, predictions):
        self.validate_predictions(predictions)
        atomic_json(self.envelope, {"version": "benchmark-checkpoint-v1", "identity": self.identity,
                                   "identity_sha256": fingerprint(self.identity), "predictions": predictions,
                                   "predictions_sha256": fingerprint(predictions)})
        atomic_json(self.export, predictions)
