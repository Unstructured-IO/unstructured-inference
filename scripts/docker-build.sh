#!/bin/bash

set -euo pipefail

DOCKER_BUILDKIT=1 docker buildx build --platform=linux/amd64 -f docker/Dockerfile \
  --progress plain \
  -t ml-inference-dev:latest .
