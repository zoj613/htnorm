#Copyright (c) 2020, Zolisa Bleki
#SPDX-License-Identifier: BSD-3-Clause */
.PHONY: clean pkg test wheels cythonize

DOCKER_IMAGES=quay.io/pypa/manylinux1_x86_64\
	      quay.io/pypa/manylinux2010_x86_64\
	      quay.io/pypa/manylinux2014_x86_64


define make_wheels
	docker pull $(1) 
	docker container run -t --rm -e PLAT=$(strip $(subst quay.io/pypa/,,$(1))) \
		-v $(shell pwd):/io $(1) /io/build-wheels.sh
endef

clean:
	rm -Rf build/* dist/* pyhtnorm/*.c pythtnorm/*.so pyhtnorm/*.html \
		__pycache__ pyhtnorm/__pycache__ pyhtnorm.egg-info

test:
	pytest -v

cythonize:
	cythonize pyhtnorm/*.pyx

sdist: cythonize
	poetry build -f sdist

wheels: clean cythonize
	$(foreach img, $(DOCKER_IMAGES), $(call make_wheels, $(img));)	
