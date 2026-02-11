import os

from sqlalchemy import create_engine, text
from sqlalchemy.exc import ProgrammingError, OperationalError


db_url = os.environ["BENCHMARK_DB_URL"]
db_username = os.environ["DB_USER"]
db_password = os.environ["DB_PASS"]


# Read-only user credentials
DB_URI = f"postgresql+psycopg://{db_username}:{db_password}@{db_url}/california_schools"
engine = create_engine(DB_URI, pool_pre_ping=True)


tests = [
    ("SELECT * FROM schools LIMIT 1;", "SELECT"),
    ("CREATE TABLE test_readonly_table2 (id SERIAL PRIMARY KEY, name TEXT)", "CREATE TABLE"),
    ("INSERT INTO schools SELECT * FROM schools LIMIT 1", "INSERT"),
    ("UPDATE schools SET County='Test' WHERE 1=1", "UPDATE"),
    ("DELETE FROM schools WHERE 1=1", "DELETE"),
    ("ALTER TABLE schools ADD COLUMN test_col INT", "ALTER TABLE")
]

with engine.connect() as conn:
    for sql, action in tests:
        print(f"=== Testing {action} ===")
        try:
            with conn.begin():  # start a fresh transaction
                result = conn.execute(text(sql))
                if action == "SELECT":
                    rows = result.fetchall()
                    for row in rows:
                        print(row)
        except Exception as e:
            if action != "SELECT":
                print(f"{action} correctly failed:", e)
            else:
                print(f"{action} incorrectly failed:", e)

