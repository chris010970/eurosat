@echo off
set GCS_BUCKET=gs://eo-ald-update
set ROOT_PATH=c:\Users\Chris.Williams\Documents\GitHub
set REPO=eurosat

REM --------------- plant type ---------------
gsutil cp %ROOT_PATH%\%REPO%\models\vgg16-256-128.zip %GCS_BUCKET%/%REPO%/models/vgg16-256-128.zip
gsutil cp %ROOT_PATH%\%REPO%\data\csv.zip %GCS_BUCKET%/%REPO%/data/csv.zip