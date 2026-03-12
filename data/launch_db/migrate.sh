#!/usr/bin/env bash
set -e

ROOT_DB_DIR="."
PG_CONTAINER="pg_benchmark" # Docker container name
PG_URI_BASE="postgresql://admin_username:admin_password@localhost:5444"
START_LETTER="a"   # Only migrate DBs starting with this letter

# Loop over all folders in ROOT_DB_DIR
for db_dir in "$ROOT_DB_DIR"/*/; do
    # Ensure it’s a directory
    [[ -d "$db_dir" ]] || continue

    db_name=$(basename "$db_dir")
    sqlite_file="$db_dir/$db_name.sqlite"

    # Skip if sqlite file doesn't exist
    [[ -f "$sqlite_file" ]] || { echo "Missing $sqlite_file, skipping"; continue; }

    # Only process directories whose first letter >= START_LETTER (case-insensitive)
    first_letter="${db_name:0:1}"
    if [[ "${first_letter,,}" < "${START_LETTER,,}" ]]; then
        echo "Skipping $db_name (before $START_LETTER)"
        continue
    fi

    echo "Processing $db_name"

    # Check if Postgres DB exists, create if missing
    exists=$(docker exec -i "$PG_CONTAINER" psql -U benchmark -d postgres -tAc "SELECT 1 FROM pg_database WHERE datname='$db_name'")
    if [[ "$exists" != "1" ]]; then
        echo "Database $db_name does not exist, creating..."
        docker exec -i "$PG_CONTAINER" psql -U benchmark -d postgres -c "CREATE DATABASE \"$db_name\""
    else
        echo "Database $db_name already exists, skipping creation"
    fi

    # pgloader call
    echo "Migrating $db_name..."
    docker run --rm \
        --network host \
        -v "$PWD:/data" \
        dimitri/pgloader:latest \
        pgloader \
        --with "batch rows = 1000" \
        --with "prefetch rows = 500" \
        "/data/$db_name/$db_name.sqlite" \
        "$PG_URI_BASE/$db_name"

    echo "Migration of $db_name completed"
done

