import os
import json
import time

from benchmarks.evaluate_bird import print_evaluation_report, sqlite_to_postgres, run_evaluation
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError, OperationalError

db_url = os.getenv("BENCHMARK_DB_URL", "localhost:5444")
results_file = os.getenv("RESULTS_FILE", "./ablation_results/feat_4_bird/query_results.json")
dataset_file = os.getenv("DATASET_FILE", "./data/bird_large.json")
checkpoint_file = results_file.replace(".json", "_eval_checkpoint.json")

db_username = os.environ["DB_USER"]
db_password = os.environ["DB_PASS"]

with open(results_file, "r") as f:
    predictions = json.load(f)
with open(dataset_file, "r") as f:
    gold_queries = {str(item["question_id"]): item for item in json.load(f)}

# Load checkpoint (already evaluated questions)
checkpoint: dict = {}
if os.path.exists(checkpoint_file):
    with open(checkpoint_file, "r") as f:
        checkpoint = json.load(f)
    print(f"Resuming from checkpoint: {len(checkpoint)}/{len(predictions)} done")
else:
    print(f"Starting fresh evaluation of {len(predictions)} predictions")

engines: dict = {}
total = len(predictions)

def get_engine(db_id: str):
    if db_id not in engines:
        uri = f"postgresql+psycopg://{db_username}:{db_password}@{db_url}/{db_id}"
        engines[db_id] = create_engine(uri)
    return engines[db_id]

for idx, (question_id, predicted_sql) in enumerate(predictions.items(), 1):
    if question_id in checkpoint:
        continue  # already evaluated

    gold_query = gold_queries[question_id]
    gold_sql = gold_query["SQL"]
    db_id = gold_query["db_id"]
    difficulty = gold_query["difficulty"]

    print(f"[{idx}/{total}] q_id={question_id} db={db_id}", flush=True)

    if gold_sql == "ambiguous" or predicted_sql == "ambiguous":
        score = 1 if gold_sql == predicted_sql == "ambiguous" else 0
        checkpoint[question_id] = {"score": score, "difficulty": difficulty,
                                   "gold_sql": gold_sql, "predicted_sql": predicted_sql}
        with open(checkpoint_file, "w") as f:
            json.dump(checkpoint, f)
        continue

    gold_sql_pg = sqlite_to_postgres(gold_sql)
    retries = 3
    for attempt in range(retries):
        try:
            engine = get_engine(db_id)
            with engine.connect() as conn:
                conn.execute(text("SET statement_timeout = '30s'"))
                gold_res = conn.execute(text(gold_sql_pg)).fetchall()
                pred_res = conn.execute(text(predicted_sql)).fetchall()
            def _to_hashable(v):
                if isinstance(v, list):
                    return tuple(_to_hashable(i) for i in v)
                if isinstance(v, dict):
                    return tuple(sorted((k, _to_hashable(val)) for k, val in v.items()))
                return v

            def _normalize(row):
                return tuple(_to_hashable(v) for v in row)

            score = int(set(_normalize(r) for r in pred_res) == set(_normalize(r) for r in gold_res))
            checkpoint[question_id] = {"score": score, "difficulty": difficulty,
                                       "gold_sql": gold_sql, "predicted_sql": predicted_sql}
            break
        except OperationalError as e:
            if attempt < retries - 1:
                wait = 30 * (attempt + 1)
                print(f"  Connection error, retrying in {wait}s... ({e})", flush=True)
                engines.pop(db_id, None)  # drop cached engine
                time.sleep(wait)
            else:
                print(f"  FAILED after {retries} attempts: {e}", flush=True)
                checkpoint[question_id] = {"score": 0, "difficulty": difficulty,
                                           "gold_sql": gold_sql, "predicted_sql": predicted_sql,
                                           "error": str(e)}
        except SQLAlchemyError as e:
            print(f"  SQL error: {e}", flush=True)
            checkpoint[question_id] = {"score": 0, "difficulty": difficulty,
                                       "gold_sql": gold_sql, "predicted_sql": predicted_sql,
                                       "error": str(e)}
            break

    with open(checkpoint_file, "w") as f:
        json.dump(checkpoint, f)

# Compute final accuracy
results = list(checkpoint.values())
total_score = sum(r["score"] for r in results)
print(f"\nOverall accuracy: {total_score / len(results) * 100:.2f}% ({total_score}/{len(results)})")

by_difficulty: dict = {}
for r in results:
    d = r.get("difficulty", "unknown")
    by_difficulty.setdefault(d, []).append(r["score"])
for diff, scores in sorted(by_difficulty.items()):
    print(f"  {diff}: {sum(scores)/len(scores)*100:.2f}% ({sum(scores)}/{len(scores)})")
