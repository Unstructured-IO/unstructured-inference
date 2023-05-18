PACKAGE_NAME := unstructured_inference
PIP_VERSION := 23.1.2


.PHONY: help
help: Makefile
	@sed -n 's/^\(## \)\([a-zA-Z]\)/\2/p' $<


###########
# Install #
###########

## install-base:            installs core requirements needed for text processing bricks
.PHONY: install-base
install-base: install-base-pip-packages
	pip install -r requirements/base.txt

## install:                 installs all test, dev, and experimental requirements
.PHONY: install
install: install-base-pip-packages install-dev install-detectron2

.PHONY: install-ci
install-ci: install-base-pip-packages install-test install-paddleocr

.PHONY: install-base-pip-packages
install-base-pip-packages:
	python3 -m pip install pip==${PIP_VERSION}

.PHONY: install-detectron2
install-detectron2:
	pip install "detectron2@git+https://github.com/facebookresearch/detectron2.git@e2ce8dc#egg=detectron2"

.PHONY: install-paddleocr
install-paddleocr:
	pip install "unstructured.PaddleOCR"

.PHONY: install-test
install-test: install-base
	pip install -r requirements/test.txt

.PHONY: install-dev
install-dev: install-test
	pip install -r requirements/dev.txt

## pip-compile:             compiles all base/dev/test requirements
.PHONY: pip-compile
pip-compile:
	pip-compile --upgrade requirements/base.in
	# NOTE(robinson) - We want the dependencies for detectron2 in the requirements.txt, but not
	# the detectron2 repo itself. If detectron2 is in the requirements.txt file, an order of
	# operations issue related to the torch library causes the install to fail
	sed 's/^detectron2 @/# detectron2 @/g' requirements/base.txt
	pip-compile --upgrade requirements/test.in
	pip-compile --upgrade requirements/dev.in


#################
# Test and Lint #
#################

## test:                    runs all unittests
.PHONY: test
test:
	PYTHONPATH=. pytest -m "not slow" test_${PACKAGE_NAME} --cov=${PACKAGE_NAME} --cov-report term-missing

.PHONY: test-slow
test-slow:
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
	scripts/version-sync.sh -c \
		-s CHANGELOG.md \
		-f unstructured_inference/__version__.py semver

## tidy:                    run black
.PHONY: tidy
tidy:
	black --line-length 100 ${PACKAGE_NAME}
	black --line-length 100 test_${PACKAGE_NAME}

## version-sync:            update __version__.py with most recent version from CHANGELOG.md
.PHONY: version-sync
version-sync:
	scripts/version-sync.sh \
		-s CHANGELOG.md \
		-f unstructured_inference/__version__.py semver

.PHONY: check-coverage
check-coverage:
	coverage report --fail-under=95
