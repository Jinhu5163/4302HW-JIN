TARGET = expx
SRC    = expx.c

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

$(TARGET): $(SRC)
	-${CLINKER} -o $(TARGET) $(SRC) ${PETSC_LIB}
