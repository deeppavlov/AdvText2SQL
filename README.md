# AdvText2SQL

This repository is made for testing student-written text2SQL tools with the help of different modern text2SQL benchmarks like BIRD.
For now there is only the BIRD benchmark integrated. Planned: integrating the Ambrosia benchmark for testing *request ambiguity*.

## Usage

First, you need to add LLM credentials. For that, copy the `.env.example` file
to `.env`:
```bash
cp .env.example .env
```
and fill out all the variables.

Now, before running the benchmark you should choose which dataset you want to use. There are two datasets:
- `data/some_queries.json` - this is about 10% of the BIRD benchmark's dev dataset. It's there, so you can save tokens and do things like debugging or estimating the efficiency of your tool as cheaply as possible. You should probably use this one at first, unless your model is free.
- `data/dev_queries.json` - this is the full BIRD benchmark's dev dataset. The baseline tool in this repo spends about 2M tokens on this.

To use a benchmark do in `bird_benchmark.py`:
```python
benchmark = BenchmarkBIRD(
    db_url=db_url,
    query_file="./data/some_queries.json",
    answer_file="./data/some_queries.json",
    use_evidence = True,
)
```

Then you will need to use your tool class and not the baseline in `bird_benchmark.py`:
```python
report = asyncio.run(benchmark.run(YourText2SQLToolClass))
```
Of course, if you want to see how baseline performs against your tool, you can switch it back to the original.

Then you can run the BIRD benchmark with:
```bash
uv run --env-file .env bird_benchmark.py
```

Additionally, after running the above command and generating the resulting queries, they are saved in a file, so you can do the evaluation again to save tokens:
```bash
uv run --env-file .env bird_evaluate_only.py
```
This could help in *debugging* (you can change the code to being more verbose in `benchmarks/evaluate.py`).

There are a few files made for debugging already (they get created during the benchmark):
- `db_schemas.json` - contains the db schemas for every database connected.
- `query_results.json` - contains generated queries.
- `all_gold_results.json` - contains the results of gold queries execution.
- `all_predicted_results.json` - contains the results of generated queries execution.

## Known issues

Since the databases are hosted remotely via PostgreSQL, there are sometimes issues translating the SQLite dialect to PostgreSQL dialect. As a result, some gold queries could produce bad results (even though we filtered them a bit), even though your query is correct. So, don't get discouraged if your accuracy is slightly lower than the public text2sql tools. The accuracy penalty should be relatively small from that, though.

## Tool requirements

As of now, you should be generating queries in the **PostgreSQL** dialect of SQL.

Your *tool class* must:
- have the `__init__` method having the following siganture: `__init__(db_url: str, llm_client: OpenAI)`. So, your tool class only needs to work with a single database at a time. It's up to you if you actually want to use the provided `llm_client`, though.
- have the `build()` method defined. This will be called only once on benchmark launch. For example, you can obtain the `db_schema` of the database in that method or add any other code you want to run during benchmark startup.
- have the `query(user_request: str) -> str` method, which intakes the user request and outputs the generated SQL query.
