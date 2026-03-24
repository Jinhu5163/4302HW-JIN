static char help[] =
"2D nonlinear reaction-diffusion problem with DMDA and SNES.\n"
"Solves  -Laplace(u) + gamma * u^p = f  on the unit square\n"
"with Dirichlet boundary condition u = u_exact.\n"
"Option prefix: -rct_\n\n";

#include <petsc.h>

typedef struct {
    PetscReal gamma;
    PetscReal p;
    PetscBool linear_f;
} AppCtx;

extern PetscReal      ufunction(PetscReal, PetscReal);
extern PetscReal      d2ufunction(PetscReal, PetscReal);
extern PetscReal      f_rhs(PetscReal, PetscReal, AppCtx*);
extern PetscErrorCode formExact(DM, Vec);
extern PetscErrorCode formF(DM, Vec, AppCtx*);
extern PetscErrorCode formRankMap(DM, Vec, PetscInt);
extern PetscErrorCode InitialGuess(DM, Vec);
extern PetscErrorCode FormFunctionLocal(DMDALocalInfo*, PetscScalar**, PetscScalar**, AppCtx*);
extern PetscErrorCode FormJacobianLocal(DMDALocalInfo*, PetscScalar**, Mat, Mat, AppCtx*);

/* ========================= MAIN ========================= */

int main(int argc,char **args)
{
    DM            da;
    SNES          snes;
    Vec           u, uexact, f, rankmap;
    PetscReal     errnorm, uexactnorm;
    DMDALocalInfo info;
    PetscMPIInt   rank;
    PetscViewer   viewer;
    AppCtx        user;

    PetscCall(PetscInitialize(&argc,&args,NULL,help));

    user.gamma    = 1.0;
    user.p        = 2.0;
    user.linear_f = PETSC_FALSE;

    PetscOptionsBegin(PETSC_COMM_WORLD,"rct_","options for reaction2d","");
    PetscCall(PetscOptionsReal("-gamma",
                               "reaction coefficient gamma",
                               "reaction2d.c",
                               user.gamma,&user.gamma,NULL));
    PetscCall(PetscOptionsReal("-p",
                               "nonlinear exponent p",
                               "reaction2d.c",
                               user.p,&user.p,NULL));
    PetscCall(PetscOptionsBool("-linear_f",
                               "if true, use f = -Laplace(u_exact); else use full nonlinear manufactured RHS",
                               "reaction2d.c",
                               user.linear_f,&user.linear_f,NULL));
    PetscOptionsEnd();

    PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));

    PetscCall(DMDACreate2d(PETSC_COMM_WORLD,
                           DM_BOUNDARY_NONE, DM_BOUNDARY_NONE,
                           DMDA_STENCIL_STAR,
                           9,9,
                           PETSC_DECIDE,PETSC_DECIDE,
                           1,1,
                           NULL,NULL,&da));
    PetscCall(DMSetFromOptions(da));
    PetscCall(DMSetUp(da));
    PetscCall(DMDASetUniformCoordinates(da,0.0,1.0,0.0,1.0,0.0,1.0));
    PetscCall(DMSetApplicationContext(da,&user));

    PetscCall(DMCreateGlobalVector(da,&u));
    PetscCall(VecDuplicate(u,&uexact));
    PetscCall(VecDuplicate(u,&f));
    PetscCall(VecDuplicate(u,&rankmap));

    PetscCall(formExact(da,uexact));
    PetscCall(InitialGuess(da,u));
    PetscCall(formF(da,f,&user));
    PetscCall(formRankMap(da,rankmap,(PetscInt)rank));

    PetscCall(SNESCreate(PETSC_COMM_WORLD,&snes));
    PetscCall(SNESSetDM(snes,da));
    PetscCall(DMDASNESSetFunctionLocal(
        da,INSERT_VALUES,
        (DMDASNESFunctionFn *)FormFunctionLocal,&user));
    PetscCall(DMDASNESSetJacobianLocal(
        da,
        (DMDASNESJacobianFn *)FormJacobianLocal,&user));
    PetscCall(SNESSetFromOptions(snes));

    PetscCall(SNESSolve(snes,NULL,u));

    PetscCall(PetscViewerVTKOpen(PETSC_COMM_WORLD,"reaction.vtr",FILE_MODE_WRITE,&viewer));
    PetscCall(PetscObjectSetName((PetscObject)uexact,"uexact"));
    PetscCall(PetscObjectSetName((PetscObject)u,"u"));
    PetscCall(PetscObjectSetName((PetscObject)f,"f"));
    PetscCall(PetscObjectSetName((PetscObject)rankmap,"rankmap"));
    PetscCall(VecView(uexact,viewer));
    PetscCall(VecView(u,viewer));
    PetscCall(VecView(f,viewer));
    PetscCall(VecView(rankmap,viewer));
    PetscCall(DMView(da,viewer));
    PetscCall(PetscViewerDestroy(&viewer));

    PetscCall(VecNorm(uexact,NORM_2,&uexactnorm));
    PetscCall(VecAXPY(u,-1.0,uexact));   /* u <- u - uexact */
    PetscCall(VecNorm(u,NORM_2,&errnorm));
    PetscCall(DMDAGetLocalInfo(da,&info));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,
              "on %d x %d grid: rel_error |u-uexact|_2/|uexact|_2 = %g\n",
              info.mx,info.my,(double)(errnorm/uexactnorm)));

    PetscCall(VecDestroy(&u));
    PetscCall(VecDestroy(&uexact));
    PetscCall(VecDestroy(&f));
    PetscCall(VecDestroy(&rankmap));
    PetscCall(SNESDestroy(&snes));
    PetscCall(DMDestroy(&da));
    PetscCall(PetscFinalize());
    return 0;
}

/* ========================= EXACT SOLUTION ========================= */

PetscReal ufunction(PetscReal x, PetscReal y)
{
    PetscReal sigma = 0.3;
    PetscReal x0    = 0.65, y0 = 0.65;
    PetscReal r2    = (x-x0)*(x-x0) + (y-y0)*(y-y0);
    PetscReal amp   = 1.0;
    return amp * PetscExpReal(-r2/(sigma*sigma));
}

PetscReal d2ufunction(PetscReal x, PetscReal y)
{
    PetscReal sigma   = 0.3;
    PetscReal x0      = 0.65, y0 = 0.65;
    PetscReal amp     = 1.0;
    PetscReal r2      = (x-x0)*(x-x0) + (y-y0)*(y-y0);
    PetscReal expterm = PetscExpReal(-r2/(sigma*sigma));

    /* returns u_xx + u_yy = Laplace(u_exact) */
    return amp * expterm * 4.0/(sigma*sigma) * (r2/(sigma*sigma) - 1.0);
}

PetscReal f_rhs(PetscReal x, PetscReal y, AppCtx *user)
{
    PetscReal uex = ufunction(x,y);
    PetscReal rhs = -d2ufunction(x,y);  /* -Laplace(u_exact) */

    if (!user->linear_f) {
        rhs += user->gamma * PetscPowReal(uex,user->p);
    }
    return rhs;
}

/* ========================= VECTOR BUILDERS ========================= */

PetscErrorCode formExact(DM da, Vec uexact)
{
    PetscInt       i, j;
    PetscReal      hx, hy, x, y, **auexact;
    DMDALocalInfo  info;

    PetscCall(DMDAGetLocalInfo(da,&info));
    hx = 1.0/(info.mx-1);
    hy = 1.0/(info.my-1);

    PetscCall(DMDAVecGetArray(da,uexact,&auexact));
    for (j=info.ys; j<info.ys+info.ym; j++) {
        y = j * hy;
        for (i=info.xs; i<info.xs+info.xm; i++) {
            x = i * hx;
            auexact[j][i] = ufunction(x,y);
        }
    }
    PetscCall(DMDAVecRestoreArray(da,uexact,&auexact));
    return 0;
}

PetscErrorCode formF(DM da, Vec f, AppCtx *user)
{
    PetscInt       i, j;
    PetscReal      hx, hy, x, y, **af;
    DMDALocalInfo  info;

    PetscCall(DMDAGetLocalInfo(da,&info));
    hx = 1.0/(info.mx-1);
    hy = 1.0/(info.my-1);

    PetscCall(DMDAVecGetArray(da,f,&af));
    for (j=info.ys; j<info.ys+info.ym; j++) {
        y = j * hy;
        for (i=info.xs; i<info.xs+info.xm; i++) {
            x = i * hx;
            af[j][i] = f_rhs(x,y,user);
        }
    }
    PetscCall(DMDAVecRestoreArray(da,f,&af));
    return 0;
}

PetscErrorCode formRankMap(DM da, Vec rankmap, PetscInt rank)
{
    PetscInt       i, j;
    PetscReal      **a;
    DMDALocalInfo  info;

    PetscCall(DMDAGetLocalInfo(da,&info));
    PetscCall(DMDAVecGetArray(da,rankmap,&a));
    for (j=info.ys; j<info.ys+info.ym; j++) {
        for (i=info.xs; i<info.xs+info.xm; i++) {
            a[j][i] = (PetscReal)rank;
        }
    }
    PetscCall(DMDAVecRestoreArray(da,rankmap,&a));
    return 0;
}

PetscErrorCode InitialGuess(DM da, Vec u)
{
    PetscInt       i, j;
    PetscReal      hx, hy, x, y, **au;
    DMDALocalInfo  info;

    PetscCall(DMDAGetLocalInfo(da,&info));
    hx = 1.0/(info.mx-1);
    hy = 1.0/(info.my-1);

    PetscCall(DMDAVecGetArray(da,u,&au));
    for (j=info.ys; j<info.ys+info.ym; j++) {
        y = j * hy;
        for (i=info.xs; i<info.xs+info.xm; i++) {
            x = i * hx;
            if (i == 0 || i == info.mx-1 || j == 0 || j == info.my-1) {
                au[j][i] = ufunction(x,y);
            } else {
                au[j][i] = 0.0;
            }
        }
    }
    PetscCall(DMDAVecRestoreArray(da,u,&au));
    return 0;
}

/* ========================= NONLINEAR RESIDUAL ========================= */

PetscErrorCode FormFunctionLocal(DMDALocalInfo *info,
                                 PetscScalar **u,
                                 PetscScalar **FF,
                                 AppCtx *user)
{
    PetscInt   i, j;
    PetscReal  hx, hy, x, y, rhs, uex;

    hx = 1.0/(info->mx-1);
    hy = 1.0/(info->my-1);

    for (j=info->ys; j<info->ys+info->ym; j++) {
        y = j * hy;
        for (i=info->xs; i<info->xs+info->xm; i++) {
            x = i * hx;

            if (i == 0 || i == info->mx-1 || j == 0 || j == info->my-1) {
                uex = ufunction(x,y);
                FF[j][i] = u[j][i] - uex;
            } else {
                rhs = f_rhs(x,y,user);

                FF[j][i] =
                    2.0*(hy/hx + hx/hy)*u[j][i]
                    - (hy/hx)*u[j][i-1]
                    - (hy/hx)*u[j][i+1]
                    - (hx/hy)*u[j-1][i]
                    - (hx/hy)*u[j+1][i]
                    + hx*hy*user->gamma*PetscPowReal(u[j][i],user->p)
                    - hx*hy*rhs;
            }
        }
    }
    return 0;
}

/* ========================= JACOBIAN ========================= */

PetscErrorCode FormJacobianLocal(DMDALocalInfo *info,
                                 PetscScalar **u,
                                 Mat J, Mat P,
                                 AppCtx *user)
{
    MatStencil row, col[5];
    PetscReal  hx, hy, v[5];
    PetscInt   i, j, ncols;

    hx = 1.0/(info->mx-1);
    hy = 1.0/(info->my-1);

    for (j=info->ys; j<info->ys+info->ym; j++) {
        for (i=info->xs; i<info->xs+info->xm; i++) {
            row.i = i;
            row.j = j;
            row.c = 0;

            if (i == 0 || i == info->mx-1 || j == 0 || j == info->my-1) {
                col[0].i = i;
                col[0].j = j;
                col[0].c = 0;
                v[0] = 1.0;
                PetscCall(MatSetValuesStencil(P,1,&row,1,col,v,INSERT_VALUES));
            } else {
                ncols = 0;

                col[ncols].i = i;
                col[ncols].j = j;
                col[ncols].c = 0;
                v[ncols++] =
                    2.0*(hy/hx + hx/hy)
                    + hx*hy*user->gamma*user->p*PetscPowReal(u[j][i],user->p - 1.0);

                col[ncols].i = i-1;
                col[ncols].j = j;
                col[ncols].c = 0;
                v[ncols++] = -hy/hx;

                col[ncols].i = i+1;
                col[ncols].j = j;
                col[ncols].c = 0;
                v[ncols++] = -hy/hx;

                col[ncols].i = i;
                col[ncols].j = j-1;
                col[ncols].c = 0;
                v[ncols++] = -hx/hy;

                col[ncols].i = i;
                col[ncols].j = j+1;
                col[ncols].c = 0;
                v[ncols++] = -hx/hy;

                PetscCall(MatSetValuesStencil(P,1,&row,ncols,col,v,INSERT_VALUES));
            }
        }
    }

    PetscCall(MatAssemblyBegin(P,MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(P,MAT_FINAL_ASSEMBLY));
    if (J != P) {
        PetscCall(MatAssemblyBegin(J,MAT_FINAL_ASSEMBLY));
        PetscCall(MatAssemblyEnd(J,MAT_FINAL_ASSEMBLY));
    }
    return 0;
}