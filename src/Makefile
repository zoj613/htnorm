.PHONY: clean lib

NAME := htnorm
CC := gcc

CFLAGS := -std=c11 -fwrapv -O3 -fPIC -funroll-loops -pthread -pedantic -g \
	-DNDEBUG -ffast-math -Wall -Wextra -Werror -Wsign-compare -Wunused \
	-Wno-unused-result -Wpointer-arith -Wcast-qual -Wmissing-prototypes \
	-Wno-missing-braces

INCLUDE_DIRS := -I. -I../include -I/usr/include

LDIR := ../lib
LIBS_DIRS := -L/usr/lib
LIBS := -lm -lcblas -llapacke

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
	$(CC) -shared -Wl,-soname=lib$(NAME) -o $@ $(LIBS) $^

clean:
	rm -f $(ODIR)/*.o  $(LDIR)/*
