# Minimal Sphinx makefile

SPHINXBUILD ?= sphinx-build
SOURCEDIR = source
BUILDDIR = _build

.PHONY: help clean html

help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)"

clean:
	rm -rf "$(BUILDDIR)"

html:
	$(SPHINXBUILD) -W --keep-going -b html "$(SOURCEDIR)" "$(BUILDDIR)/html"
