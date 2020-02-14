init:
	git submodule update --init
	cd submodules/sylph && git submodule update --init

black:
	black --config black.toml .

lint:
	flake8 elaenia tests
	mypy --check-untyped-defs --config-file=tox.ini elaenia

test:
	pytest --ignore submodules --ignore sylph/vendor

ipython:
	ipython -i scripts/python_session_init.py
