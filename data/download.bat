@echo off
set GCS_BUCKET=gs://eo-ald-update
set REPO=eurosat

gsutil cp %GCS_BUCKET%/%REPO%/data/csv.zip .
