PACKAGE_NAME := unstructured_inference
PIP_VERSION := 22.3


.PHONY: help
help: Makefile
	@sed -n 's/^\(## \)\([a-zA-Z]\)/\2/p' $<


###########
# Install #
###########

## install-base:            installs core requirements needed for text processing bricks
.PHONY: install-base
install-base: install-base-pip-packages

## install:                 installs all test, dev, and experimental requirements
.PHONY: install
install: install-base-pip-packages install-dev install-detectron2 install-test

.PHONY: install-ci
install-ci: install-base-pip-packages install-test

.PHONY: install-base-pip-packages
install-base-pip-packages:
	python3 -m pip install pip==${PIP_VERSION}
	pip install -r requirements/base.txt

.PHONY: install-detectron2
install-detectron2:
	pip install "detectron2@git+https://github.com/facebookresearch/detectron2.git@v0.6#egg=detectron2"

.PHONY: install-test
install-test:
	pip install -r requirements/test.txt

.PHONY: install-dev
install-dev:
	pip install -r requirements/dev.txt

## pip-compile:             compiles all base/dev/test requirements
.PHONY: pip-compile
pip-compile:
	pip-compile requirements/base.in
	# NOTE(robinson) - We want the dependencies for detectron2 in the requirements.txt, but not
	# the detectron2 repo itself. If detectron2 is in the requirements.txt file, an order of
	# operations issue related to the torch library causes the install to fail
	sed 's/^detectron2 @/# detectron2 @/g' requirements/base.txt
	pip-compile requirements/dev.in
	pip-compile requirements/test.in

#########
# Build #
#########

## docker-build:            builds the docker container for detectron2
.PHONY: docker-build
docker-build:
	PIP_VERSION=${PIP_VERSION}  ./scripts/docker-build.sh

#########
# Local #
########

## download-models          downloads unstructured models (AWS credentials must be in environment variables)
.PHONY: download-models
download-models:
	./scripts/dl-models.sh

## run-app-dev:             runs the FastAPI api with hot reloading
.PHONY: run-app-dev
run-app-dev:
	PYTHONPATH=. uvicorn unstructured_inference.api:app --reload

## start-app-local:         runs FastAPI in the container with hot reloading
.PHONY: start-app-local
start-app-local:
	docker run --name=ml-inference-container -p 127.0.0.1:5000:5000 ml-inference-dev

## stop-app-local:          stops the container
.PHONY: stop-app-local
stop-app-local:
	docker stop ml-inference-container | xargs docker rm

#################
# Test and Lint #
#################

## test:                    runs all unittests
.PHONY: test
test:
	PYTHONPATH=. pytest test_${PACKAGE_NAME} --cov=${PACKAGE_NAME} --cov-report term-missing

## check:                   runs linters (includes tests)
.PHONY: check
check: check-src check-tests check-version

## check-src:               runs linters (source only, no tests)
.PHONY: check-src
check-src:
	black --line-length 100 ${PACKAGE_NAME} --check
	flake8 ${PACKAGE_NAME}
	mypy ${PACKAGE_NAME} --ignore-missing-imports

.PHONY: check-tests
check-tests:
	black --line-length 100 test_${PACKAGE_NAME} --check
	flake8 test_${PACKAGE_NAME}

## check-scripts:           run shellcheck
.PHONY: check-scripts
check-scripts:
    # Fail if any of these files have warnings
	scripts/shellcheck.sh

## check-version:           run check to ensure version in CHANGELOG.md matches version in package
.PHONY: check-version
check-version:
    # Fail if syncing version would produce changes
	scripts/version-sync.sh -c

## tidy:                    run black
.PHONY: tidy
tidy:
	black --line-length 100 ${PACKAGE_NAME}
	black --line-length 100 test_${PACKAGE_NAME}

## version-sync:            update __version__.py with most recent version from CHANGELOG.md
.PHONY: version-sync
version-sync:
	scripts/version-sync.sh

.PHONY: check-coverage
check-coverage:
	coverage report --fail-under=95
