#include <petsc.h>
#include <math.h>

int main(int argc, char **argv)
{
  PetscMPIInt rank, size;
  PetscReal   x = 1.0;
  PetscInt    N = 10;

  PetscCall(PetscInitialize(&argc, &argv, NULL,
                            "Compute exp(x) in parallel with PETSc.\n\n"));

  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
  PetscOptionsBegin(PETSC_COMM_WORLD, "", "options for expx", "");
  PetscCall(PetscOptionsReal("-x", "input to exp(x) function", NULL, x, &x, NULL));
  PetscCall(PetscOptionsInt ("-N", "number of Taylor terms",   NULL, N, &N, NULL));
  PetscOptionsEnd();

  if (N < 1) SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "N must be >= 1");
  PetscInt k0 = (rank * N) / size;
  PetscInt k1 = ((rank + 1) * N) / size;

  PetscReal term     = 1.0;
  PetscReal localsum = 0.0;

  for (PetscInt k = 1; k <= k0; k++) term *= x / (PetscReal)k;

  for (PetscInt k = k0; k < k1; k++) {
    localsum += term;
    term *= x / (PetscReal)(k + 1);
  }

  PetscReal globalsum = 0.0;
  PetscCallMPI(MPI_Allreduce(&localsum, &globalsum, 1, MPIU_REAL, MPIU_SUM, PETSC_COMM_WORLD));


  if (rank == 0) {
    PetscReal exact  = (PetscReal)exp((double)x);
    PetscReal relerr = PetscAbsReal(globalsum - exact) / PetscAbsReal(exact);
    PetscReal mult   = relerr / PETSC_MACHINE_EPSILON;

    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "x=%g  N=%d  nP=%d\n", (double)x, (int)N, (int)size));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "approx     = %.16e\n", (double)globalsum));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "exact      = %.16e\n", (double)exact));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "relerr     = %.3e\n",  (double)relerr));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "relerr/eps = %.3e\n",  (double)mult));
  }

  PetscCall(PetscFinalize());
  return 0;
}

