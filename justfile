# Development tasks for intset. Run `just` to list them.

set positional-arguments

default:
    @just --list

# Install the development environment
sync:
    uv sync --locked --all-groups

# Run the test suite (pass extra pytest arguments after --)
test *args:
    uv run pytest tests "$@"

# Run the test suite with many more Hypothesis examples, as CI does
test-thorough *args:
    HYPOTHESIS_PROFILE=ci uv run pytest tests "$@"

# Run the tests with coverage, failing if any line or branch is missed
coverage:
    uv run pytest tests --cov=intset --cov-report=term-missing

# Run the benchmark suite
bench *args:
    uv run pytest bench -p benchmark --benchmark-warmup=on "$@"

# Run the benchmark suite as ordinary tests, without timing anything
bench-check:
    uv run pytest bench -p benchmark --benchmark-disable

# Lint with ruff
lint:
    uv run ruff check src tests bench source scripts
    uv run ruff format --check src tests bench source scripts

# Reformat the code and fix what ruff can fix automatically
format:
    uv run ruff check --fix src tests bench source scripts
    uv run ruff format src tests bench source scripts

# Build the HTML documentation into build/html
docs:
    uv run sphinx-build -W -b html source build/html

# Build the sdist and wheel into dist/
build:
    rm -rf dist
    uv build
    uv run --isolated --no-project --with dist/*.whl python -c "import intset; print('built intset', intset.__version__)"

# Run everything that CI runs
check: lint coverage bench-check docs build

# Print the current version
version:
    @uv run python scripts/release.py version

# Cut a release: bump the version, stamp the changelog, tag and push. CI then publishes to PyPI.
release version:
    #!/usr/bin/env bash
    set -euo pipefail
    if [ -n "$(git status --porcelain)" ]; then
        echo "Working tree is not clean; commit or stash your changes first." >&2
        exit 1
    fi
    if [ "$(git rev-parse --abbrev-ref HEAD)" != "master" ]; then
        echo "Releases must be cut from master." >&2
        exit 1
    fi
    git fetch origin master
    if [ "$(git rev-parse HEAD)" != "$(git rev-parse origin/master)" ]; then
        echo "Local master is not in sync with origin/master." >&2
        exit 1
    fi
    uv run python scripts/release.py prepare "$1"
    just check
    git add CHANGELOG.rst src/intset/version.py
    git commit -m "Release $1"
    git tag -a "v$1" -m "Release $1"
    git push origin master "v$1"
    echo "Pushed v$1. Watch the release workflow with: gh run watch"

# Publish dist/ to PyPI from this machine (fallback if the release workflow cannot)
publish: build
    uv publish
