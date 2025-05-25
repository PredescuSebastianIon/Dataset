.PHONY: all venv install activate run clean

VENV := .venv
PIP := pip
PYTHON := python3

all: activate install run

venv:
	@echo "Creating virtual environment in root folder..."
	python3 -m venv $(VENV)

install: venv
	@echo "Installing dependencies..."
	$(PIP) install -r requirements.txt

activate:
	@echo "To activate the virtual environment, run:"
	@echo "source $(VENV)/bin/activate"

run:
	cd src && $(MAKE)

clean:
	@echo "Cleaning virtual environment and caches..."
	rm -rf $(VENV)
	cd src && $(MAKE) clean

