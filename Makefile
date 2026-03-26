VENV = .venv/bin/python

.PHONY: all analysis paper clean

all: analysis paper

analysis:
	$(VENV) src/analysis.py

paper:
	cd paper && latexmk -pdf -interaction=nonstopmode -synctex=1 main.tex supplementary.tex

clean:
	# -C tells latexmk to clean up all generated files (pdf, aux, logs, etc)
	cd paper && latexmk -C main.tex supplementary.tex
	rm -f paper/figures/*.pdf