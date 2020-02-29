VENV_DIR=.venv/elaenia
ASSETS_DIR=assets
SENTINELS_DIR=.make-sentinel-files
$(shell mkdir -p $(ASSETS_DIR) $(SENTINELS_DIR))


init: submodules virtualenv python-packages vggish_checkpoint_file
	@echo To activate the submodule and set environment variables:
	@echo source env.sh


lint:
	flake8 elaenia tests
	mypy --check-untyped-defs --config-file=tox.ini elaenia


test:
	pytest \
		--doctest-modules \
		--ignore .venv \
		--ignore submodules \
		--ignore sylph/vendor


submodules:
	git submodule update --init
	cd submodules/sylph && git submodule update --init


eggs: submodules elaenia-egg sylph-egg


elaenia-egg:
	python setup.py bdist_egg


SYLPH_VENDOR_DIR=submodules/sylph/sylph/vendor
SYLPH_SUBMODULES_DIR=submodules/sylph/submodules
TF_MODELS_SYMLINK_NAME=tensorflow_models
TF_MODELS_SUBMODULE_NAME=tensorflow-models
TF_MODELS_SYMLINK=$(SYLPH_VENDOR_DIR)/$(TF_MODELS_SYMLINK_NAME)
TF_MODELS_SUBMODULE=$(SYLPH_SUBMODULES_DIR)/$(TF_MODELS_SUBMODULE_NAME)
sylph-egg:
	@# I don\'t know how to follow symlinks when creating an egg.
	rm $(TF_MODELS_SYMLINK)
	mv $(TF_MODELS_SUBMODULE) $(TF_MODELS_SYMLINK)
	cd submodules/sylph && python setup.py bdist_egg
	mv $(TF_MODELS_SYMLINK) $(TF_MODELS_SUBMODULE)
	cd $(SYLPH_VENDOR_DIR) && ln -s ../../submodules/$(TF_MODELS_SUBMODULE_NAME)/ $(TF_MODELS_SYMLINK_NAME)


virtualenv: $(VENV_DIR)
$(VENV_DIR):
	python_version=$$(python -c 'import sys; print("%d.%d" % sys.version_info[:2])') && \
		[ $$python_version = 3.6 ] || [ $$python_version = 3.7 ] || (echo "Python 3.6 or 3.7 required." 1>&2; exit 1 )
	mkdir -p $(VENV_DIR)
	[ -e $(VENV_DIR)/bin ] || python -m venv $(VENV_DIR)


PYTHON_PACKAGES_SENTINEL=$(SENTINELS_DIR)/python-packages
python-packages: $(PYTHON_PACKAGES_SENTINEL)
$(PYTHON_PACKAGES_SENTINEL):
	$(VENV_DIR)/bin/pip install -e .
	$(VENV_DIR)/bin/pip install -r requirements.txt -r requirements_local.txt && touch $(PYTHON_PACKAGES_SENTINEL)


VGGISH_CHECKPOINT_FILE=$(ASSETS_DIR)/vggish_model.ckpt
vggish_checkpoint_file: $(VGGISH_CHECKPOINT_FILE)
$(VGGISH_CHECKPOINT_FILE):
	curl -L https://storage.googleapis.com/audioset/vggish_model.ckpt -o $(VGGISH_CHECKPOINT_FILE)


black:
	black --config black.toml .


ipython:
	ipython -i scripts/python_session_init.py


.PHONY: init lint test submodules eggs elaenia-egg sylph-egg virtualenv python-packages vggish_checkpoint_file black ipython
