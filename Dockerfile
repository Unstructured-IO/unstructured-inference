# syntax=docker/dockerfile:experimental
FROM python:3.12-slim AS base

# Set up environment
ENV HOME=/home/
WORKDIR ${HOME}
RUN mkdir ${HOME}/.ssh && chmod go-rwx ${HOME}/.ssh \
  && ssh-keyscan -t rsa github.com >> /home/.ssh/known_hosts
ENV PYTHONPATH="${PYTHONPATH}:${HOME}"

# Install uv
COPY --from=ghcr.io/astral-sh/uv:0.6.2 /uv /uvx /usr/local/bin/

FROM base AS deps
# Copy project files needed for dependency resolution
COPY pyproject.toml uv.lock .python-version ./
COPY unstructured_inference/__version__.py unstructured_inference/__version__.py

RUN uv sync --frozen --all-groups --no-install-project

# Ensure venv binaries are on PATH so pytest/etc. are directly accessible
ENV PATH="/home/.venv/bin:${PATH}"

FROM deps AS code
COPY unstructured_inference unstructured_inference

CMD ["/bin/bash"]
