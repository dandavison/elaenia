black:
	black --config black.toml .

lint:
	flake8 elaenia tests
	mypy --check-untyped-defs --config-file=tox.ini elaenia

test:
	pytest --ignore=vendor

ipython:
	ipython -i scripts/python_session_init.py


.PHONY: lint test ipython
