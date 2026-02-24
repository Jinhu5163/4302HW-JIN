
PETSC_DIR  ?= /Users/jinhu/petsc_build/petsc
PETSC_ARCH ?= arch-macos-hdf5-debug


include $(PETSC_DIR)/lib/petsc/conf/variables
include $(PETSC_DIR)/$(PETSC_ARCH)/lib/petsc/conf/petscvariables
include $(PETSC_DIR)/lib/petsc/conf/rules


bvp: bvp.o
	-$(CLINKER) -o bvp bvp.o $(PETSC_LIB)
	$(RM) bvp.o

run:
	mpiexec -np 1 ./bvp -options_file options_file.txt


run4:
	mpiexec -np 4 ./bvp -options_file options_file.txt


clean::
	$(RM) bvp bvp.o bvp_solution.h5