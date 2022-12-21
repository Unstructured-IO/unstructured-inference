#!/bin/bash

set -euo pipefail

aws s3 cp s3://utic-dev-models/oer_checkbox/detectron2_oer_checkbox.json "${1:-.models}"/detectron2_oer_checkbox.json
aws s3 cp s3://utic-dev-models/oer_checkbox/detectron2_finetuned_oer_checkbox.pth  "${1:-.models}"/detectron2_finetuned_oer_checkbox.pth
