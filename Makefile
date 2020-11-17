.PHONY: clean pkg

clean:
	rm -Rf build/ dist/ pyhtnorm/*.c pyhtnorm/*.html __pycache__ pyhtnorm/__pycache__

pkg: clean
	poetry build -f wheel
	poetry build -f sdist
