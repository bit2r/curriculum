.PHONY: all clean install run requirements help

PYTHON=python3
PIP=pip3

SRC=penguins_pipeline.py
REQ=requirements.txt

all: install run

install:
	$(PIP) install -r $(REQ)

run:
	$(PYTHON) $(SRC)

requirements:
	$(PIP) install pipreqs
	pipreqs --force .

clean:
	rm -rf __pycache__ *.pyc
	rm results.txt
	rm requirements.txt

help:
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@echo "  help		  Show this help message"
	@echo "  all		   Install dependencies and run the script"
	@echo "  requirements  Generate requirements.txt using pipreqs"
	@echo "  install	   Install dependencies"
	@echo "  run		   Run the script"
	@echo "  clean		 Clean up the project directory"

	