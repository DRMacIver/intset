.PHONY: clean documentation


DEVELOPMENT_DATABASE?=postgres://whereshouldilive@localhost/whereshouldilive_dev
SPHINXBUILD   = $(DEV_PYTHON) -m sphinx
SPHINX_BUILDDIR      = docs/_build
ALLSPHINXOPTS   = -d $(SPHINX_BUILDDIR)/doctrees docs -W

BUILD_RUNTIMES?=$(PWD)/.runtimes

PY26=$(BUILD_RUNTIMES)/snakepit/python2.6
PY27=$(BUILD_RUNTIMES)/snakepit/python2.7
PY33=$(BUILD_RUNTIMES)/snakepit/python3.3
PY34=$(BUILD_RUNTIMES)/snakepit/python3.4
PY35=$(BUILD_RUNTIMES)/snakepit/python3.5
PYPY=$(BUILD_RUNTIMES)/snakepit/pypy

TOOLS=$(BUILD_RUNTIMES)/tools

TOX=$(TOOLS)/tox
SPHINX_BUILD=$(TOOLS)/sphinx-build
SPHINX_AUTOBUILD=$(TOOLS)/sphinx-autobuild
ISORT=$(TOOLS)/isort
FLAKE8=$(TOOLS)/flake8
PYFORMAT=$(TOOLS)/pyformat

TOOL_VIRTUALENV=$(BUILD_RUNTIMES)/virtualenvs/tools
ISORT_VIRTUALENV=$(BUILD_RUNTIMES)/virtualenvs/isort
TESTMON_VIRTUALENV=$(BUILD_RUNTIMES)/virtualenvs/testmon
TOOL_PYTHON=$(TOOL_VIRTUALENV)/bin/python
TOOL_PIP=$(TOOL_VIRTUALENV)/bin/pip
TOOL_INSTALL=$(TOOL_PIP) install --upgrade
export PATH:=$(BUILD_RUNTIMES)/snakepit:$(TOOLS):$(PATH)
export LC_ALL=C.UTF-8

$(PY26):
	scripts/install.sh 2.6

$(PY27):
	scripts/install.sh 2.7

$(PY33):
	scripts/install.sh 3.3

$(PY34):
	scripts/install.sh 3.4

$(PY35):
	scripts/install.sh 3.5

$(PYPY):
	scripts/install.sh pypy

$(TOOL_VIRTUALENV): $(PY34)
	$(PY34) -m virtualenv $(TOOL_VIRTUALENV)
	mkdir -p $(TOOLS)

$(TESTMON_VIRTUALENV): $(PY35)
	$(PY35) -m virtualenv $(TESTMON_VIRTUALENV)

testmon: $(TESTMON_VIRTUALENV)
	$(TESTMON_VIRTUALENV)/bin/python -m pip install --upgrade setuptools wheel pip
	$(TESTMON_VIRTUALENV)/bin/python -m pip install --upgrade pytest flaky pytest-testmon
	PYTHONPATH=src $(TESTMON_VIRTUALENV)/bin/python -m pytest --testmon tests/cover

$(TOOLS): $(TOOL_VIRTUALENV)

$(ISORT_VIRTUALENV): $(PY34)
	$(PY34) -m virtualenv $(ISORT_VIRTUALENV)

format: $(PYFORMAT) $(ISORT)
	$(TOOL_PYTHON) scripts/enforce_header.py
	# isort will sort packages differently depending on whether they're installed
	$(ISORT_VIRTUALENV)/bin/python -m pip install pytest
	env -i PATH=$(PATH) $(ISORT) -p hypothesis -ls -m 2 -w 75 \
			-a  "from __future__ import absolute_import, print_function, division" \
			-rc src tests 
	find src tests -name '*.py' | xargs $(PYFORMAT) -i

lint: $(FLAKE8)
	$(FLAKE8) src tests --exclude=compat.py,test_reflection.py,test_imports.py,tests/py2 --ignore=E731,E721

check-format: format lint
	git diff --exit-code

check-py26: $(PY26) $(TOX)
	$(TOX) -e py26-nocoverage

check-py27: $(PY27) $(TOX)
	$(TOX) -e py27-nocoverage

check-py33: $(PY33) $(TOX)
	$(TOX) -e py33-nocoverage

check-py34: $(py34) $(TOX)
	$(TOX) -e py34-nocoverage

check-py35: $(PY35) $(TOX)
	$(TOX) -e py35-nocoverage

check-pypy: $(PYPY) $(TOX)
	$(TOX) -e pypy-nocoverage

check-coverage: $(TOX) $(PY35)
	$(TOX) -e py35-coverage

check: check-format check-coverage check-py26 check-py27 check-py33 check-py34 check-py35 check-pypy check-django check-pytest

check-fast: lint $(PY26) $(PY35) $(PYPY) $(TOX)
	$(TOX) -e pypy-brief
	$(TOX) -e py35-brief
	$(TOX) -e py26-brief
	$(TOX) -e py35-prettyquick

$(TOX): $(PY35) tox.ini $(TOOLS)
	$(TOOL_INSTALL) tox
	rm -f $(TOX)
	rm -rf .tox
	ln -sf $(TOOL_VIRTUALENV)/bin/tox $(TOX)

$(SPHINX_BUILD): $(TOOL_VIRTUALENV)
	$(TOOL_PYTHON) -m pip install sphinx
	ln -sf $(TOOL_VIRTUALENV)/bin/sphinx-build $(SPHINX_BUILD)

$(SPHINX_AUTOBUILD): $(TOOL_VIRTUALENV)
	$(TOOL_PYTHON) -m pip install sphinx-autobuild
	ln -sf $(TOOL_VIRTUALENV)/bin/sphinx-autobuild $(SPHINX_AUTOBUILD)

$(PYFORMAT): $(TOOL_VIRTUALENV)
	$(TOOL_INSTALL) pyformat
	ln -sf $(TOOL_VIRTUALENV)/bin/pyformat $(PYFORMAT)

$(ISORT): $(ISORT_VIRTUALENV)
	$(ISORT_VIRTUALENV)/bin/python -m pip install isort==4.1.0
	ln -sf $(ISORT_VIRTUALENV)/bin/isort $(ISORT)

$(FLAKE8): $(TOOL_VIRTUALENV)
	$(TOOL_INSTALL) flake8
	ln -sf $(TOOL_VIRTUALENV)/bin/flake8 $(FLAKE8)

clean:
	rm -rf .tox
	rm -rf .hypothesis
	rm -rf docs/_build
	rm -rf $(TOOLS)
	rm -rf $(BUILD_RUNTIMES)/snakepit
	rm -rf $(BUILD_RUNTIMES)/virtualenvs
	find src tests -name "*.pyc" -delete
	find src tests -name "__pycache__" -delete

documentation: $(SPHINX_BUILD) source/*.rst
	$(SPHINX_BUILD) -W -b html -d build/doctrees   source build/html
