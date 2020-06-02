@echo off
set GCS_BUCKET=gs://eo-ald-update
set REPO=eurosat

gsutil cp %GCS_BUCKET%/%REPO%/models/vgg16-256-128.zip .
