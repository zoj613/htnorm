# copyright (c) 2020, zolisa bleki
#
# SPDX-License-Identifier: BSD-3-Clause 
.PHONY: clean lib

NAME := htnorm
CC := gcc

CFLAGS := -std=c11 -fwrapv -O3 -fPIC -funroll-loops -pedantic -g -pthread \
	-DNDEBUG -ffast-math -Wall -Wextra -Werror -Wsign-compare -Wunused \
	-Wno-unused-result -Wpointer-arith -Wcast-qual -Wmissing-prototypes \
	-Wno-missing-braces

# set default include directory for BLAS include files
INCLUDE_DIRS ?= /usr/include
override INCLUDE_DIRS := -I../include -I$(INCLUDE_DIRS)
LDIR := ../lib
# set default include directory for BLAS shared library
LIBS_DIR ?= /usr/lib
override LIBS_DIR := -L$(LIBS_DIR)
LIBS ?= -lopenblas
override LIBS += -lm

SRCFILES = $(wildcard *.c)

ODIR := obj
OBJ= $(patsubst %.c, $(ODIR)/%.o, $(SRCFILES))


$(ODIR)/%.o: %.c
	mkdir -p $(ODIR) 
	$(CC) $(CFLAGS) $(INCLUDE_DIRS) -o $@ -c $^

lib: $(LDIR)/lib$(NAME).so
	ldconfig -v -n $(LDIR)
	# dont forget to append LDIR to LD_LIBRARY_PATH

$(LDIR)/lib$(NAME).so: $(OBJ)
	mkdir -p $(LDIR)
	$(CC) -shared -Wl,-soname=lib$(NAME) -o $@ $(LIBS_DIR) $(LIBS) $^

clean:
	rm -f $(ODIR)/*.o  $(LDIR)/*
