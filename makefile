SHELL := /bin/bash

# Common settings
CHPL_COMPILER := chpl
CHPL_COMMON_OPTS := --fast -M lib/common

# Source files
CHPL_NQUEENS_SOURCES := nqueens_chpl.chpl nqueens_gpu_chpl.chpl nqueens_multigpu_chpl.chpl
CHPL_PFSP_SOURCES := pfsp_chpl.chpl pfsp_gpu_chpl.chpl pfsp_multigpu_chpl.chpl pfsp_dist_multigpu_chpl.chpl

# Object files
CHPL_NQUEENS_OBJECTS := $(CHPL_NQUEENS_SOURCES:.chpl=.o)
CHPL_PFSP_OBJECTS := $(CHPL_PFSP_SOURCES:.chpl=.o)

# Library paths
CHPL_NQUEENS_LIBPATH := -M lib/nqueens
CHPL_PFSP_LIBPATH := -M lib/pfsp

# Build codes
all: $(CHPL_NQUEENS_OBJECTS) $(CHPL_PFSP_OBJECTS)

# N-Queens

nqueens_chpl.o: nqueens_chpl.chpl
	$(CHPL_COMPILER) $(CHPL_COMMON_OPTS) $(CHPL_NQUEENS_LIBPATH) $< -o $@

nqueens_gpu_chpl.o: nqueens_gpu_chpl.chpl
	$(CHPL_COMPILER) $(CHPL_COMMON_OPTS) $(CHPL_NQUEENS_LIBPATH) $< -o $@

nqueens_multigpu_chpl.o: nqueens_multigpu_chpl.chpl
	$(CHPL_COMPILER) $(CHPL_COMMON_OPTS) $(CHPL_NQUEENS_LIBPATH) $< -o $@

nqueens_dist_multigpu_chpl.o: nqueens_dist_multigpu_chpl.chpl
	$(CHPL_COMPILER) $(CHPL_COMMON_OPTS) $(CHPL_NQUEENS_LIBPATH) $< -o $@

# PFSP

pfsp_chpl.o: pfsp_chpl.chpl
	$(CHPL_COMPILER) $(CHPL_COMMON_OPTS) $(CHPL_PFSP_LIBPATH) $< -o $@

pfsp_gpu_chpl.o: pfsp_gpu_chpl.chpl
	$(CHPL_COMPILER) $(CHPL_COMMON_OPTS) $(CHPL_PFSP_LIBPATH) $< -o $@

pfsp_multigpu_chpl.o: pfsp_multigpu_chpl.chpl
	$(CHPL_COMPILER) $(CHPL_COMMON_OPTS) $(CHPL_PFSP_LIBPATH) $< -o $@

pfsp_dist_multigpu_chpl.o: pfsp_dist_multigpu_chpl.chpl
	$(CHPL_COMPILER) $(CHPL_COMMON_OPTS) $(CHPL_PFSP_LIBPATH) $< -o $@

# Utilities
.PHONY: clean

clean:
	rm -f $(CHPL_NQUEENS_OBJECTS) $(CHPL_PFSP_OBJECTS) *_real
