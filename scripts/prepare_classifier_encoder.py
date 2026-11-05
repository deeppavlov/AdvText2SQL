"""Download a pinned frozen encoder into the workspace; never contact the SQL LLM."""

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_OUTPUT = ROOT / ".cache/classifier/minilm"
FILES = ("config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json",
         "special_tokens_map.json", "vocab.txt")


def verify_encoder(folder):
    manifest = json.loads((folder / "encoder.json").read_text(encoding="utf-8"))
    if manifest.get("format_version") != "frozen-encoder-v1" or not manifest["files_sha256"]:
        raise ValueError("Invalid encoder manifest")
    for name, expected in manifest["files_sha256"].items():
        if name not in FILES or hashlib.sha256((folder / name).read_bytes()).hexdigest() != expected:
            raise ValueError("Frozen encoder file changed")
    if not {"config.json", "model.safetensors", "tokenizer.json"} <= set(manifest["files_sha256"]):
        raise ValueError("Incomplete encoder")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--download", action="store_true", help="Allow the Hugging Face download")
    args = parser.parse_args()
    output = args.out.resolve()
    if not output.is_relative_to(ROOT / ".cache/classifier"):
        parser.error("Encoder cache must stay inside .cache/classifier/")
    if (output / "encoder.json").exists():
        info = verify_encoder(output)
        if info["model"] != args.model or args.revision not in ("main", info["revision"]):
            parser.error("Existing encoder differs; choose a new cache folder")
        print(f"VERIFIED: {info['model']} @ {info['revision']}: {output}")
        return
    if not args.download:
        parser.error("Encoder not prepared. Use --download once to allow the model download")
    from huggingface_hub import HfApi, snapshot_download
    revision = HfApi().model_info(args.model, revision=args.revision).sha
    snapshot_download(args.model, revision=revision, local_dir=output, allow_patterns=list(FILES))
    hashes = {name: hashlib.sha256((output / name).read_bytes()).hexdigest() for name in FILES if (output / name).exists()}
    info = {"format_version": "frozen-encoder-v1", "model": args.model, "revision": revision,
            "pooling": "attention-masked mean, L2 normalization", "max_length": 256, "files_sha256": hashes}
    (output / "encoder.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8", newline="\n")
    verify_encoder(output)
    print(f"PREPARED: {args.model} @ {revision}: {output}")


if __name__ == "__main__":
    main()
