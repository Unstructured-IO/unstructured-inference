#!/usr/bin/env bash

set -euo pipefail
PIP_VERSION="${PIP_VERSION:-23.1.2}"
DOCKER_IMAGE="unstructured-inference:dev"

DOCKER_BUILD_CMD=(docker buildx build --load -f Dockerfile \
  --build-arg PIP_VERSION="$PIP_VERSION" \
  --build-arg BUILDKIT_INLINE_CACHE=1 \
  --progress plain \
  -t "$DOCKER_IMAGE" .)

DOCKER_BUILDKIT=1 "${DOCKER_BUILD_CMD[@]}"