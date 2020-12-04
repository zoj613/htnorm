#Copyright (c) 2020, Zolisa Bleki
#SPDX-License-Identifier: BSD-3-Clause */
.PHONY: clean pkg test wheels cythonize lib

NAME := htnorm
CC := gcc

CFLAGS := -std=c11 -fwrapv -O3 -fPIC -funroll-loops -pedantic -g -pthread \
	-DNDEBUG -ffast-math -Wall -Wextra -Werror -Wsign-compare -Wunused \
	-Wno-unused-result -Wpointer-arith -Wcast-qual -Wmissing-prototypes \
	-Wno-missing-braces

# set default include directory for BLAS include files
INCLUDE_DIR ?= /usr/include
override INCLUDE_DIR := -I./include -I$(INCLUDE_DIR)
LDIR := ./lib
# set default include directory for BLAS shared library
LIBS_DIR ?= /usr/lib
override LIBS_DIR := -L$(LIBS_DIR)
LIBS ?= -lopenblas
override LIBS += -lm

_SRCFILES = $(wildcard src/*.c)
SRCFILES = $(filter-out src/r_wrapper.c, $(_SRCFILES))

OBJ= $(patsubst %.c, %.o, $(SRCFILES))


%.o: %.c
	$(CC) $(CFLAGS) $(INCLUDE_DIR) -o $@ -c $^

lib: $(LDIR)/lib$(NAME).so
	ldconfig -v -n $(LDIR)
	# dont forget to append LDIR to LD_LIBRARY_PATH

$(LDIR)/lib$(NAME).so: $(OBJ)
	mkdir -p $(LDIR)
	$(CC) -shared -Wl,-soname=lib$(NAME) -o $@ $(LIBS_DIR) $(LIBS) $^
	rm $(OBJ)

clean:
	rm -Rf $(LDIR)/* build/* dist/* pyhtnorm/*.c pythtnorm/%.so \
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
