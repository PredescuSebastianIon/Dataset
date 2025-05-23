PYTHON := python3
SRCS := main.py preprocess.py

.PHONY: all preprocess run force rerun

all: preprocess run

preprocess:
	@echo "Proprocessing the dataset"
	$(PYTHON) preprocess.py

run: preprocess
	@echo "Starting server"
	$(PYTHON) main.py

force:

rerun: force
	@echo "Rerunning"
	$(MAKE) run
