.PHONY: install clean example activate

VENV = venv
PYTHON = $(VENV)/bin/python
PIP = $(VENV)/bin/pip
PROJECT_ROOT = $(shell pwd)
FREEQDSK_DIR = $(PROJECT_ROOT)/gysmc/freeqdsk
EXAMPLES_DIR = $(PROJECT_ROOT)/examples

install: $(VENV) submodules
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	$(PIP) install fortranformat~=2.0
	$(PIP) install -e $(FREEQDSK_DIR)
	@if ! grep -q "export PYTHONPATH.*$(PROJECT_ROOT)" $(VENV)/bin/activate; then \
		echo "export PYTHONPATH=\"$$PYTHONPATH:$(PROJECT_ROOT)\"" >> $(VENV)/bin/activate; \
	fi
	@echo "Installation complete! Activate the virtual environment with: source $(VENV)/bin/activate"

submodules:
	@if [ -f .gitmodules ]; then \
		git submodule update --init --recursive; \
	fi

$(VENV):
	python3 -m venv $(VENV)

example:
	@mkdir -p $(EXAMPLES_DIR)
	@if [ ! -f $(EXAMPLES_DIR)/g031213.00003 ]; then \
		echo "Downloading example GEQDSK file..."; \
		curl -L -o $(EXAMPLES_DIR)/g031213.00003 https://pwl.home.ipp.mpg.de/NLED_AUG/g031213.00003; \
		echo "Downloaded to $(EXAMPLES_DIR)/g031213.00003"; \
	else \
		echo "Example file already exists: $(EXAMPLES_DIR)/g031213.00003"; \
	fi
	@if [ ! -f $(EXAMPLES_DIR)/params_gvec_W7X.ini ]; then \
		echo "Downloading example W7X equilibrium file..."; \
		curl -L -o $(EXAMPLES_DIR)/params_gvec_W7X.ini https://gitlab.mpcdf.mpg.de/gvec-group/gvec/-/raw/develop/test-CI/examples/w7x/parameter.ini?ref_type=heads; \
		echo "Downloaded to $(EXAMPLES_DIR)/params_gvec_W7X.ini"; \
	else \
		echo "Example file already exists: $(EXAMPLES_DIR)/params_gvec_W7X.ini"; \
	fi

activate:
	@if [ ! -d $(VENV) ]; then \
		echo "Virtual environment not found. Run 'make install' first."; \
		exit 1; \
	fi
	@echo "To activate the virtual environment, run:"
	@echo "  source $(VENV)/bin/activate"
	@echo ""
	@echo "Or use the following command:"
	@echo "  . $(VENV)/bin/activate"

clean:
	rm -rf $(VENV)
	@echo "Virtual environment removed"

