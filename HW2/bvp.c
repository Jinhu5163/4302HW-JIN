
static char help[] =
"Solve 1D BVP: -u''(x) + gamma u(x) = f(x) on [0,1] with Dirichlet BCs\n"
"using PETSc. Manufactured solution:\n"
"  u(x) = sin(k*pi*x) + c*(x-1/2)^3\n"
"Options:\n"
"  -bvp_m <int>      (grid parameter, h = 1/m)\n"
"  -bvp_gamma <real>\n"
"  -bvp_k <int>\n"
"  -bvp_c <real>\n";

#include <petsc.h>
#include <petscviewerhdf5.h>

int main(int argc, char **args)
{
    Vec         u, f, uexact, err;
    Mat         A;
    KSP         ksp;
    PetscViewer viewer;

    PetscInt    m = 201;     
    PetscInt    n;           
    PetscInt    k = 5;       
    PetscInt    i, Istart, Iend;
    PetscInt    j[3];

    PetscReal   gamma = 0.0;
    PetscReal   c = 3.0;
    PetscReal   h, x, t;
    PetscReal   pi = PETSC_PI;
    PetscReal   kpi, sinpart;
    PetscReal   uval, fval;
    PetscReal   vals[3];
    PetscReal   errnorm, unorm, relerr;

    PetscScalar diag;
    PetscInt    bcrows[2];

    PetscCall(PetscInitialize(&argc, &args, NULL, help));

    
    PetscOptionsBegin(PETSC_COMM_WORLD, NULL, "options for bvp", NULL);
    PetscCall(PetscOptionsInt("-bvp_m", "grid parameter m so h=1/m", "bvp.c", m, &m, NULL));
    PetscCall(PetscOptionsReal("-bvp_gamma", "gamma in -u'' + gamma u = f", "bvp.c", gamma, &gamma, NULL));
    PetscCall(PetscOptionsInt("-bvp_k", "k in sin(k*pi*x)", "bvp.c", k, &k, NULL));
    PetscCall(PetscOptionsReal("-bvp_c", "c in c*(x-1/2)^3", "bvp.c", c, &c, NULL));
    PetscOptionsEnd();

    if (m < 2) SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "-bvp_m must be >= 2");
    if (k < 1) SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "-bvp_k must be >= 1");

    h   = 1.0 / (PetscReal)m;
    n   = m + 1;           
    kpi = ((PetscReal)k) * pi;


    PetscCall(VecCreate(PETSC_COMM_WORLD, &u));
    PetscCall(VecSetSizes(u, PETSC_DECIDE, n));
    PetscCall(VecSetFromOptions(u));
    PetscCall(VecDuplicate(u, &f));
    PetscCall(VecDuplicate(u, &uexact));
    PetscCall(VecDuplicate(u, &err));


    PetscCall(MatCreate(PETSC_COMM_WORLD, &A));
    PetscCall(MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, n, n));
    PetscCall(MatSetOptionsPrefix(A, "a_"));
    PetscCall(MatSetFromOptions(A));
    PetscCall(MatSetUp(A));

    PetscCall(MatGetOwnershipRange(A, &Istart, &Iend));
    for (i = Istart; i < Iend; i++) {
        if (i == 0) {
            vals[0] = 2.0/(h*h) + gamma;
            vals[1] = -1.0/(h*h);
            j[0]    = 0;
            j[1]    = 1;
            PetscCall(MatSetValues(A, 1, &i, 2, j, vals, INSERT_VALUES));
        } else if (i == n-1) {
            vals[0] = -1.0/(h*h);
            vals[1] = 2.0/(h*h) + gamma;
            j[0]    = i-1;
            j[1]    = i;
            PetscCall(MatSetValues(A, 1, &i, 2, j, vals, INSERT_VALUES));
        } else {
            vals[0] = -1.0/(h*h);
            vals[1] =  2.0/(h*h) + gamma;
            vals[2] = -1.0/(h*h);
            j[0]    = i-1;
            j[1]    = i;
            j[2]    = i+1;
            PetscCall(MatSetValues(A, 1, &i, 3, j, vals, INSERT_VALUES));
        }
    }


    PetscCall(VecGetOwnershipRange(uexact, &Istart, &Iend));
    for (i = Istart; i < Iend; i++) {
        x       = ((PetscReal)i) * h;
        t       = x - 0.5;
        sinpart = PetscSinReal(kpi * x);

        uval = sinpart + c * t * t * t;
        fval = ((kpi * kpi) + gamma) * sinpart - 6.0 * c * t + gamma * c * t * t * t;

        PetscCall(VecSetValue(uexact, i, (PetscScalar)uval, INSERT_VALUES));
        PetscCall(VecSetValue(f,      i, (PetscScalar)fval, INSERT_VALUES));
    }

    PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));
    PetscCall(VecAssemblyBegin(uexact));
    PetscCall(VecAssemblyEnd(uexact));
    PetscCall(VecAssemblyBegin(f));
    PetscCall(VecAssemblyEnd(f));

   
    bcrows[0] = 0;
    bcrows[1] = n - 1;
    diag      = 1.0;
    PetscCall(MatZeroRowsColumns(A, 2, bcrows, diag, uexact, f));

    PetscCall(KSPCreate(PETSC_COMM_WORLD, &ksp));
    PetscCall(KSPSetOperators(ksp, A, A));
    PetscCall(KSPSetFromOptions(ksp));
    PetscCall(KSPSolve(ksp, f, u));

    PetscCall(VecCopy(u, err));
    PetscCall(VecAXPY(err, -1.0, uexact));
    PetscCall(VecNorm(err, NORM_2, &errnorm));
    PetscCall(VecNorm(uexact, NORM_2, &unorm));
    relerr = (unorm > 0.0) ? (errnorm / unorm) : errnorm;

    PetscCall(PetscPrintf(PETSC_COMM_WORLD,
        "BVP solve: m=%d (n=%d), h=%.6e, gamma=%g, k=%d, c=%g\n",
        (int)m, (int)n, (double)h, (double)gamma, (int)k, (double)c));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,
        "Absolute error ||u-uexact||_2 = %.12e\n", (double)errnorm));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,
        "Relative error ||u-uexact||_2 / ||uexact||_2 = %.12e\n", (double)relerr));

PetscCall(PetscViewerHDF5Open(PETSC_COMM_WORLD, "bvp_solution.h5", FILE_MODE_WRITE, &viewer));
PetscCall(PetscObjectSetName((PetscObject)uexact, "uexact"));
PetscCall(PetscObjectSetName((PetscObject)f,      "f"));
PetscCall(PetscObjectSetName((PetscObject)u,      "u"));
PetscCall(VecView(f, viewer));
PetscCall(VecView(u, viewer));
PetscCall(VecView(uexact, viewer));
PetscCall(PetscViewerDestroy(&viewer));

    PetscCall(KSPDestroy(&ksp));
    PetscCall(MatDestroy(&A));
    PetscCall(VecDestroy(&u));
    PetscCall(VecDestroy(&f));
    PetscCall(VecDestroy(&uexact));
    PetscCall(VecDestroy(&err));
    PetscCall(PetscFinalize());
    return 0;
}
