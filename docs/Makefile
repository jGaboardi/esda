# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line.
SPHINXOPTS    =
SPHINXBUILD   = sphinx-build
SPHINXPROJ    = PACKAGE_NAME
SOURCEDIR     = .
BUILDDIR      = build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@rsync -r --exclude '.ipynb_checkpoints/' ../notebooks/ ./notebooks/
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

github:
	@make html

sync:

	@rsync -avh --exclude '.nojekyll' build/html/ ../docs/ --delete
	@make clean
	touch ../docs/.nojekyll

clean:
	rm -rf $(BUILDDIR)/*
	rm -rf auto_examples/
	rm -rf generated/
