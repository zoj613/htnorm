.PHONY: clean pkg test

clean:
	rm -Rf build/ dist/ pyhtnorm/*.c pyhtnorm/*.html __pycache__ pyhtnorm/__pycache__

test:
	pytest -v

pkg: clean
	poetry build -f wheel
	poetry build -f sdist
