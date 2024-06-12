run:
	cd docs && bundle exec jekyll serve

build:
	rm -rf _site
	cd docs && bundle exec jekyll build && mv _site ../_site

