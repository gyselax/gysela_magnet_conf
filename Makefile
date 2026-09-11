.PHONY: install install_full clean example activate

VENV = venv
PYTHON = $(VENV)/bin/python
PIP = $(VENV)/bin/pip
PROJECT_ROOT = $(shell pwd)
EXAMPLES_DIR = $(PROJECT_ROOT)/examples

install: $(VENV)
	$(PIP) install .
	@echo "Installation complete! Activate the virtual environment with: source $(VENV)/bin/activate"

install_full: $(VENV)
	$(PIP) install .[geqdsk,gvec]
	@echo "Installation complete! Activate the virtual environment with: source $(VENV)/bin/activate"

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
