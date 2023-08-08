#!/usr/bin/env bash

# This is intended to be run from an unstructured checkout, not in this repo
# The goal here is to see what changes the current branch would introduce to unstructured
# fixtures

# Define your commands in an array
INGEST_COMMANDS=(
    test_unstructured_ingest/test-ingest-azure.sh
    test_unstructured_ingest/test-ingest-biomed-api.sh
    test_unstructured_ingest/test-ingest-biomed-path.sh
    test_unstructured_ingest/test-ingest-box.sh
    test_unstructured_ingest/test-ingest-dropbox.sh
    test_unstructured_ingest/test-ingest-gcs.sh
    test_unstructured_ingest/test-ingest-onedrive.sh
    test_unstructured_ingest/test-ingest-s3.sh
)

# An array to store exit statuses
EXIT_STATUSES=()

# Run each command and capture its exit status
for INGEST_COMMAND in "${INGEST_COMMANDS[@]}"; do
  $INGEST_COMMAND
  EXIT_STATUSES+=($?)
done

# Check for failures
for STATUS in "${EXIT_STATUSES[@]}"; do
  if [[ $STATUS -ne 0 ]]; then
    echo "At least one ingest command failed! Scroll up to see which"
    exit 1
  fi
done

echo "No diff's resulted from any ingest commands"
