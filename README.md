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
## Installing the BIRD dev database

For doing the BIRD benchmark, you can install their dev dataset from their official website (1.5 GB).


## Installing the Spider 1.0 Database

For doing the Spider 1.0 benchmark you'll need to install the public Spider Database (1.8 GB).
You can either download it manually from [this link](https://drive.google.com/file/d/1403EGqzIDoHMdQF4c9Bkyl7dZLZ5Wt6J/view?usp=sharing), in that case, remember to place it into the `data`.

Or you can run the following script in the `data` directory (`curl` didn't work for me, cause it's Google Drive):
```bash
pip install gdown
gdown https://drive.google.com/uc?id=1403EGqzIDoHMdQF4c9Bkyl7dZLZ5Wt6J
```
Then unzip the installed archive:
```bash
unzip spider_data.zip
```
You can remove the archive to save some space:
```bash
rm spider_data.zip
```

## Text2SQL tool

Before launching the `text2sql_tool` which is already in this repo, you need to set LLM credentials in the `.env` file, namely `LLM_BASE_URL` and `LLM_API_KEY`.

Dev note: It's a working MCP tool, but proper integration with the PydanticAI `Agent` isn't done yet. For now, it will be called as a normal Python function.

## Benchmark and Evaluation

The baseline tool will be available for launch with the spider_benchmark.py file. Instructions on how to run your tool instead will be provided shortly.
This repo is using the `Spider 1.0` benchmark. A migration to `UNITE` or `Spider 2.0` will be possible later, in case we want a more robust benchmark, the APIs are mostly similar.
