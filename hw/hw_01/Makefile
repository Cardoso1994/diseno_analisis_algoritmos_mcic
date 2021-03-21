###############################################################################
#
# Tareas Clasificación Inteligente de Patrones
#
###############################################################################

HW = hw_final
REF = referencias.bib

SLIDES = slides

IMG = ./img


all: $(HW).pdf

$(HW).pdf: $(HW).tex $(REF)
	@ pdflatex -shell-escape $(HW).tex
	@ bibtex $(HW)
	@ pdflatex -shell-escape $(HW).tex
	@ pdflatex -shell-escape $(HW).tex


clean:
	@ rm -rf *.aux *.bbl *.blg *.lof *.log *.lot *.out *.toc
