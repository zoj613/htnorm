#Copyright (c) 2020, Zolisa Bleki
#SPDX-License-Identifier: BSD-3-Clause */
.PHONY: clean pkg test wheels cythonize lib

NAME := htnorm
CC := gcc

CFLAGS := -std=c11 -fwrapv -O3 -fPIC -funroll-loops -pedantic -g -pthread \
	-DNDEBUG -ffast-math -Wall -Wextra -Werror -Wsign-compare -Wunused \
	-Wno-unused-result -Wpointer-arith -Wcast-qual -Wmissing-prototypes \
	-Wno-missing-braces -Wstrict-aliasing -fstrict-aliasing -Winline

INCLUDE_DIR := -I./include
LDIR := ./lib
LIBS_DIR ?= /usr/lib
override LIBS_DIR := -L$(LIBS_DIR)
LIBS := -lm -lblas -llapack

SRCFILES = src/dist.c src/htnorm.c src/rng.c

OBJ = src/dist.o src/htnorm.o src/rng.o


%.o: %.c
	$(CC) $(CFLAGS) $(INCLUDE_DIR) -o $@ -c $^

lib: $(LDIR)/lib$(NAME).so
	ldconfig -v -n $(LDIR)
	# dont forget to append LDIR to LD_LIBRARY_PATH

$(LDIR)/lib$(NAME).so: $(OBJ)
	mkdir -p $(LDIR)
	$(CC) -pthread -shared -Wl,-soname=lib$(NAME) -o $@ $(LIBS_DIR) $(LIBS) $^
	rm $(OBJ)

clean:
	rm -Rf $(LDIR)/* build/* dist/* pyhtnorm/*.c pyhtnorm/*.so \
		pyhtnorm/*.html pyhtnorm.egg-info src/*.o **/*__pycache__ \
		**/*_snaps ..Rcheck src/*.so src/*.rds __pycache__


# python dev targets
# ==================
DOCKER_IMAGES=quay.io/pypa/manylinux1_x86_64 \
	      quay.io/pypa/manylinux2010_x86_64 \
	      quay.io/pypa/manylinux2014_x86_64

define make_wheels
	docker pull $(1) 
	docker container run -t --rm -e PLAT=$(strip $(subst quay.io/pypa/,,$(1))) \
		-v $(shell pwd):/io $(1) /io/build-wheels.sh
endef

test:
	pytest -v

cythonize:
	cythonize pyhtnorm/*.pyx

sdist: cythonize
	poetry build -f sdist

wheels: clean cythonize
	$(foreach img, $(DOCKER_IMAGES), $(call make_wheels, $(img));)	


# R dev targets
# ============
check:
	R CMD check --no-manual --no-vignettes --timings .
