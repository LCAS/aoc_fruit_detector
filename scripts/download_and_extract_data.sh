#!/usr/bin/env bash

set -euo pipefail

URL="https://lcas.lincoln.ac.uk/nextcloud/index.php/s/sXpCCQDeafD6Qto/download"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="$SCRIPT_DIR/data"
ARCHIVE="$DATA_DIR/download"

if [[ ! -d "$DATA_DIR/fruit_detector" ]]; then
    mkdir -p "$DATA_DIR"

    if [[ ! -s "$ARCHIVE" ]]; then
        echo "This may take a few minutes..."
        curl --fail --location --show-error --progress-bar "$URL" --output "$ARCHIVE"
    else
        echo "Archive already downloaded; skipping download."
    fi

    unzip -o "$ARCHIVE" -d "$DATA_DIR"
else
    echo "Dataset folder already exists; skipping download and extraction."
fi

while IFS= read -r -d '' ZIP_FILE; do
    ZIP_DIR="$(dirname -- "$ZIP_FILE")"
    echo "Extracting $ZIP_FILE..."
    unzip -o "$ZIP_FILE" -d "$ZIP_DIR"
    rm -- "$ZIP_FILE"
done < <(find "$DATA_DIR" -type f -name '*.zip' -print0)
