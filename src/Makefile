PYTHON := python3
SRCS := main.py

.PHONY: all run force rerun

all: run

run:
	@echo "Starting server"
	$(PYTHON) main.py

force:

rerun: force
	@echo "Rerunning"
	$(MAKE) run

clean:
	@echo "→ Cleaning up…"
	rm -rf __pycache__ */__pycache__ *.pyc */*.pyc
	rm -rf .venv
	rm -f ../data/train.csv ../data/test.csv ../data/diabetes.csv
	rm -rf ../figures
