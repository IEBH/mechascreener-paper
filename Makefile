# Define the python interpreter
VENV = .venv/bin/python

# TELL MAKE THESE ARE COMMANDS, NOT FILES
# This fixes the conflict where Make thinks the "paper" folder is the build result.
.PHONY: all analysis paper clean

all: analysis paper

# Run analysis
analysis:
	$(VENV) src/analysis.py

# Build paper
paper:
	cd paper && pdflatex main.tex
	cd paper && bibtex main
	cd paper && pdflatex main.tex
	cd paper && pdflatex main.tex

# Cleanup
clean:
	rm -f paper/*.aux paper/*.log paper/*.pdf paper/*.bbl paper/*.blg
	rm -f paper/figures/*.pdf