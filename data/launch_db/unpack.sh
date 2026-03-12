#!/usr/bin/env bash

set -e

TARGET_DIR="$1"

if [ -z "$TARGET_DIR" ]; then
    echo "Usage: $0 <target_directory>"
    exit 1
fi

mkdir -p "$TARGET_DIR"

for domain in */; do
    [ -d "$domain" ] || continue

    for dbdir in "$domain"*/; do
        [ -d "$dbdir" ] || continue

        basename=$(basename "$dbdir")
        destination="$TARGET_DIR/$basename"

        if [ -e "$destination" ]; then
            echo "Directory already exists: $basename"
            continue
        fi

        mv "$dbdir" "$destination"
    done
done
