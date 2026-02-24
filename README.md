# AdvText2SQL

This repository is made for testing student-written text2SQL tools with the help of different modern text2SQL benchmarks like BIRD and Ambrosia. BIRD benchmark tests SQL generation, while the Ambrosia benchmark tests for how a tool reacts to ambiguity.

## Getting started

First, you need to add LLM credentials. For that, copy the `.env.example` file
to `.env`:
```bash
cp .env.example .env
```
and fill out all the variables.

Before running a benchmark you should choose which dataset you want to use. There are two datasets for the BIRD benchmark:
- `data/some_queries.json` (22 questions) - this is a small part of the BIRD benchmark's dev dataset. It's there, so you can save tokens and do things like debugging or estimating the efficiency of your tool as cheaply as possible. You should probably use this one at first, unless your model is free.
- `data/dev_queries.json` (241 questions) - this is the full BIRD benchmark's dev dataset. The baseline tool in this repo spends about 2M tokens on this.

The Ambrosia benchmark checks how your tool reacts to ambiguity. It has two testing datasets: `ambrosia_dev` (240 questions) and `ambrosia_small` (24 questions)

To choose which BIRD benchmark dataset you're going to use, set the right file in the `bird_benchmark.py`:
```python
benchmark = BenchmarkBIRD(
    db_url=db_url,
    query_file="./data/some_queries.json",
    answer_file="./data/some_queries.json",
    use_evidence = True,
)
```
*"evidence"* is some additional data which might be useful to the LLM. For example, it could help decipher an abbreviation for the LLM model. Evidence usually gets appended to the `user_request`. Only available in the BIRD benchmark, I suggest to use it.

Then you can run the BIRD benchmark with:
```bash
uv run --env-file .env bird_benchmark.py
```
After also choosing the dataset for Ambrosia run:
```bash
uv run --env-file .env ambrosia_benchmark.py
```

Additionally, after running the BIRD benchmark and generating queries, they are all saved in a file, so you can do the evaluation again to save tokens:
```bash
uv run --env-file .env bird_evaluate_only.py
```
This could help in debugging the benchmark (you can change the code to being more verbose in `benchmarks/evaluate.py`) or if you want to reuse yesterday's results without spending a million of tokens.

There are also a few files made for debugging (they get created during both benchmarks):
- `db_schemas.json` - contains the collected db schemas for every database connected.
- `query_results.json` - contains your generated queries.

BIRD only logs:
- `all_gold_results.json` - contains the results of gold queries execution.
- `all_predicted_results.json` - contains the results of generated queries execution.

## Writing your own tool

First, create your own `git` branch to store your tool code. You can create your own tool straight in the `text2sql_implementation.py` file, which contains the baseline tool. Please, don't make any commits to the `main` branch.

You are free to use the baseline tool as an inspiration, but it's actually not that great (only 30% accuracy). Moreover, one of the goals of the course is to beat this exact baseline tool, so you'll probably need to rewrite that anyway.

Just make sure that your resulting tool class still has these methods:
- the `__init__` method having the following signature: `__init__(db_url: str, llm_client: OpenAI)`. Basically, your tool class only needs to work with a single database at a time. It's up to you if you actually want to use the provided `llm_client`, though.
- the `build()` method. This method will be called once for every tool instance at the start of a benchmark run. For example, you can obtain the `db_schema` of the databases within that method or add any other code you want to run during benchmark startup.
- the `query(user_request: str) -> dict[str, Any]` method, which intakes the user request and outputs the generated SQL query. Output format:
```python
{
    "status": "success",
    "query": some_sql_query
}
```

In the real world some requests are going to be ambiguous - either too vague to create a query, out-of-scope of the database or maybe the request has several possible interpretations which affect the answer. Carefully consider how you will handle those queries, because the `query()` method must be ready for ambiguous requests. 

In case the user request is ambiguous, the tool should set `"status": "ambiguous"`.

Note that the generated queries must use the **PostgreSQL** dialect of SQL. Also, please, prepare your queries for being launched immediately - remove markdown blocks, validate the sql syntax. In case there is an error in your sql query, the score for that question will be zero.

## Some suggestions

Here are a few suggestions for improving your tools. Be careful with some of them, you could just waste tokens or bloat the context if not careful.

- For starters, you should probably add the "db_schema", some "unique values" and maybe "row samples" into the model prompt. Without them the model won't know how the db looks.

- During the baseline tool's benchmarking, the model would often hallucinate non-existent table and column names. Consider trying to execute the generated query, and giving the result to the LLM for checking if it matches the `user_request` or if there were any errors. Then, the LLM will be given a chance to adjust it's answer if something's wrong.
The process should repeat until there have been too many attempts, like 3-5. (or you'll just run out of tokens, probably)

- During the baseline tool's benchmarking, the model would often perform poorly in terms of Postgres syntax. In particular, there were some quirks with `"SELECT DISTINCT"` and `"ORDER BY"` clauses that `sqlglot` validator passed as valid queries, but in reality they weren't. I think it could be a good idea to look closely at such spots where your model fails, then add more rules to your prompt, until the model is able to recognize those syntax rules.

- Ambiguity checking seems to be rather difficult for `gpt-3.5-turbo`, and LLMs in general. In particular, beware of false positives - `gpt-3.5-turbo` would mark up to 50% of completely valid requests as ambiguous. It would probably be best to do some research on ambiguity checking, before starting to work on that feature.

## Known issues

Since the databases are hosted remotely via PostgreSQL, there are sometimes issues translating the SQLite dialect to PostgreSQL dialect. As a result, some gold queries could produce bad results (even though we filtered them a bit), even though your tool's query is correct. So, don't get discouraged if your accuracy is slightly lower than the public text2sql tools. The accuracy penalty should be relatively small from that, though.

## Baseline tool results

As of now, the average accuracy of the baseline tool using gpt-3.5-turbo is somewhere around 20-35% (didn't check the larger dataset, plus it's different for every benchmark run). This accuracy is quite low, but the official BIRD benchmark's website reports that this is close to the expected accuracy of just using ChatGPT with nothing else.

At the same time, ambiguity false positives on the BIRD dataset are around 20%.

Ambiguity accuracy is in general hard to estimate, but results of running the small ambrosia dataset were around 50-65%.

## Training

There are also two training datasets in this repo:
- `data/train_queries.json` - this is the BIRD benchmark's train dataset. It has questions labeled by difficulty and gold queries for each question. It has no ambiguous questions.
- `data/ambrosia_train.json` - this is the Ambrosia benchmark's train dataset. It has both ambiguous and non-ambiguous questions. The file contains pairs `question` -> `ambig_type`, which you can use to train the tool. Secondly, there are `ambig_question` and `ambig_queries` fields for each `question_id`. Basically, it's `ambiguous_question` -> `possible queries`. So, it's like an additional dataset of ambiguous queries. You should check it out yourself to really get it.

## Links

We're using the BIRD benchmark's `dev` dataset, which can be found [here](https://bird-bench.github.io/).
BIRD is an industrial-grade text2sql benchmark with cases that are close to real-world queries.

We're also using [Ambrosia](https://ambrosia-benchmark.github.io/) - a text2sql benchmark with a main focus on ambiguous requests.
