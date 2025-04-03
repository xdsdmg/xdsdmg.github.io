NAME = ai-share.tex

build: ${NAME}
	xelatex ${NAME}
	biber ai-share
	xelatex ${NAME}
	xelatex ${NAME}

clean:
	rm -rf *.pdf *.aux *.log *.out *.xml *.bcf

.PHONY: build clean