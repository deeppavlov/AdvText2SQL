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
*"evidence"* is some additional data which might be useful to the LLM. For example, it could help decipher an abbreviation for the LLM model. Evidence usually gets appended to the `user_request`.

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
- `query_results.json` - contains your generated queries.
- `all_gold_results.json` - contains the results of gold queries execution.
- `all_predicted_results.json` - contains the results of generated queries execution.

## Known issues

Since the databases are hosted remotely via PostgreSQL, there are sometimes issues translating the SQLite dialect to PostgreSQL dialect. As a result, some gold queries could produce bad results (even though we filtered them a bit), even though your query is correct. So, don't get discouraged if your accuracy is slightly lower than the public text2sql tools. The accuracy penalty should be relatively small from that, though.

## Baseline tool results

As of now, the average accuracy of the baseline tool using gpt-3.5-turbo is somewhere around 25-35% (didn't check the larger dataset, plus it's different for every benchmark run). This accuracy is quite low, but the official BIRD benchmark's website reports that this is close to the expected accuracy of just using ChatGPT with nothing else.

## Tool requirements

First, create your own `git` branch to store your tool code. You can create your own tool straight in the `text2sql_implementation.py` file, which contains the baseline tool. Please, don't make any commits to the `main` branch.

You are free to use the baseline tool as an inspiration, but it's actually not that great (only 30% accuracy). Moreover, one of the goals of the course is to beat this exact baseline tool, so you'll probably need to rewrite that anyway.

Just make sure that your resulting tool class still has these methods:
- the `__init__` method having the following signature: `__init__(db_url: str, llm_client: OpenAI)`. Basically, your tool class only needs to work with a single database at a time. It's up to you if you actually want to use the provided `llm_client`, though.
- the `build()` method. This method will be called on benchmark launch. For example, you can obtain the `db_schema` of the database within that method or add any other code you want to run during benchmark startup.
- the `query(user_request: str) -> dict[str, Any]` method, which intakes the user request and outputs the generated SQL query.

In the real world some requests are going to be ambiguous - either too vague to create a query, out-of-scope of the database or maybe the request has several possible interpretations which affect the answer. Carefully consider how you will handle those queries, because the `query()` method must be ready for ambiguous requests. Check the answer schema to learn how to mark the request as ambiguous.

Note that the generated queries must use the **PostgreSQL** dialect of SQL. Also, please, prepare your queries for being launched immediately - remove markdown blocks, validate the sql syntax. In case there is an error in your sql query, the score for that question will be zero.

## Some suggestions

Here are a few suggestions for improving your tools. Be careful with some of them, you could just waste tokens or bloat the context if not careful.

- For starters, you should probably add the "db_schema", some "unique values" and maybe "row samples" into the model prompt. Without them the model won't know how the db looks.

- During the baseline tool's benchmarking, the model would often hallucinate non-existent table and column names. Consider adding some sort of recursive validators, like, an additional llm_client dialogue, which only compares the generated sql to the database_schema for correctness. Then, if it fails, it should make a more realistic suggestion to the calling agent, so that it can try again with the newly-gained knowledge. The process should repeat until there have been too many attempts, like 3-5. (or you'll run out of tokens very quickly, which would be quite bad)

- During the baseline tool's benchmarking, the model would often perform poorly in terms of Postgres syntax. In particular, there were some quirks with `"SELECT DISTINCT"` and `"ORDER BY"` clauses that `sqlglot` validator passed as valid queries, but in reality they weren't. I think it could be a good idea to look closely at such spots where your model fails, then add more rules to your prompt, until the model is able to recognize those syntax rules.

- Ambiguity checking seems to be rather difficult for `gpt-3.5-turbo`, and LLMs in general. In particular, beware of false positives - `gpt-3.5-turbo` would mark up to 50% of completely valid requests as ambiguous. It would probably be best to do some research on ambiguity checking, before starting to work on that feature.

## Links

We're using the BIRD benchmark's `dev` dataset, which can be found [here](https://bird-bench.github.io/).
BIRD is an industrial-grade text2sql benchmark with cases that are close to real-world queries.

We're also planning to use [Ambrosia](https://ambrosia-benchmark.github.io/) - a text2sql benchmark with a main focus on ambiguous requests.