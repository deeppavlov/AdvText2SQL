# Partial MCP

This repository is for testing methods to reduce LLM context occupied by
MCP tool descriptions.

## Usage

First, install pre-commit hooks with:
```bash
uv run pre-commit install
```
This will make it so you cannot make a git commit if it does not satisfy
conditions from [.pre-commit-config.yaml](.pre-commit-config.yaml). Namely:
- Large files added
- Bad formatting
- Ruff's linter fails
- Pyrefly's type checking fails

Next, you need to add LLM credentials. For that, copy the `.env.example` file
to `.env`:
```bash
cp .env.example .env
```
and fill out all the variables.

Finally, run the benchmark with:
```bash
uv run --env-file .env benchmark.py
```

If you want to chat with the agent, run the web client via:
```bash
uv run --env-file .env uvicorn partial_mcp.web:app --port 8000
```
