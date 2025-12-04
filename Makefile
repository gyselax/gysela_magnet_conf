.PHONY: install clean example

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

clean:
	rm -rf $(VENV)
	@echo "Virtual environment removed"

