// 2-D lid-driven cavity, incompressible Navier-Stokes, SMAC (Simplified Marker-And-Cell) method
// Original author: Karthick J (OE23S018), AM5630 Computer Assignment 3.
// The original program kept this structure (constants, compute_pressure, update_velocity,
// compute_vorticity, time loop, output files) but the function bodies did not solve the
// Navier-Stokes equations (see report, Section 3). The bodies were COMPLETED here following the
// SMAC algorithm stated in the original report; all completed parts are marked [COMPLETED]/[FIX].
//
// Grid: staggered MAC grid with NC x NC cells (NC = NX-1): p at cell centres, u on vertical
// faces, v on horizontal faces, one ghost layer for the wall conditions.
// SMAC step:  (1) u* = u^n + dt[-(u.grad)u + (1/Re) lap(u) - grad p^n]
//             (2) lap(phi) = div(u*)/dt   (Neumann BC, SOR)
//             (3) u^{n+1} = u* - dt grad(phi),   p^{n+1} = p^n + phi

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// Define constants
#define NX 100              // grid points per side (as original); NC = NX-1 = 99 cells
#define NY 100
#define NC (NX - 1)
#define NT 200000           // [FIX: max time steps; was 1000 (t = 1), far from steady state]
#define DX (1.0 / (NX - 1)) // [FIX: brackets added; '1.0 / (NX - 1)' without them breaks in expressions]
#define DY (1.0 / (NY - 1))
#define DT 0.001
#define RE 100.0
#define U_INFTY 1.0
#define STEADY_TOL 1.0e-7   // [ADDED] steady state: max|u^{n+1}-u^n|/U < tol
#define SOR_OMEGA 1.85      // [ADDED] SOR factor for the pressure-correction Poisson equation
#define SOR_TOL 1.0e-8      // [ADDED]

// Define variables  [FIX: double precision; float cannot converge the Poisson solve to 1e-8]
double u[NC + 1][NC + 2], v[NC + 2][NC + 1], p[NC + 2][NC + 2];
double unew[NC + 1][NC + 2], vnew[NC + 2][NC + 1], pnew[NC + 2][NC + 2];   // u*, v*, phi
double S[NC + 1][NC + 1], Vor[NC + 1][NC + 1];   // [FIX: were VLAs sized by uninitialised i, j]

// [COMPLETED] wall conditions through ghost cells: no slip on all walls, u = U_INFTY on the lid
void apply_velocity_bc(double uu[NC + 1][NC + 2], double vv[NC + 2][NC + 1]) {
    for (int i = 0; i <= NC; i++) {
        uu[i][0] = -uu[i][1];                       // bottom wall: u = 0 at y = 0
        uu[i][NC + 1] = 2.0 * U_INFTY - uu[i][NC];  // top lid:     u = U at y = 1
    }
    for (int j = 0; j <= NC + 1; j++) { uu[0][j] = 0.0; uu[NC][j] = 0.0; }   // left/right walls
    for (int j = 0; j <= NC; j++) {
        vv[0][j] = -vv[1][j];                       // left wall:  v = 0 at x = 0
        vv[NC + 1][j] = -vv[NC][j];                 // right wall: v = 0 at x = 1
    }
    for (int i = 0; i <= NC + 1; i++) { vv[i][0] = 0.0; vv[i][NC] = 0.0; }   // bottom/top walls
}

// Function to compute pressure with boundary conditions
// [COMPLETED] SMAC pressure correction: lap(phi) = div(u*)/DT, dphi/dn = 0 on walls (SOR)
int compute_pressure() {
    int i, j, it;
    for (i = 0; i <= NC + 1; i++) for (j = 0; j <= NC + 1; j++) pnew[i][j] = 0.0;
    for (it = 1; it <= 20000; it++) {
        double res = 0.0;
        for (i = 1; i <= NC; i++) {
            for (j = 1; j <= NC; j++) {
                double div = (unew[i][j] - unew[i - 1][j]) / DX + (vnew[i][j] - vnew[i][j - 1]) / DY;
                double aw = (i > 1) ? 1.0 : 0.0, ae = (i < NC) ? 1.0 : 0.0;   // Neumann walls:
                double as = (j > 1) ? 1.0 : 0.0, an = (j < NC) ? 1.0 : 0.0;   // drop wall links
                double ap = (aw + ae) / (DX * DX) + (as + an) / (DY * DY);
                double rhs = (aw * pnew[i - 1][j] + ae * pnew[i + 1][j]) / (DX * DX)
                           + (as * pnew[i][j - 1] + an * pnew[i][j + 1]) / (DY * DY) - div / DT;
                double r = rhs / ap - pnew[i][j];
                pnew[i][j] += SOR_OMEGA * r;
                if (fabs(r) > res) res = fabs(r);
            }
        }
        if (res < SOR_TOL) break;
    }
    return it;
}

// Function to update velocity with boundary conditions
// [COMPLETED] (a) predictor u*, v* from the momentum equations (explicit Euler, central differences)
void predict_velocity() {
    int i, j;
    for (i = 1; i < NC; i++) {
        for (j = 1; j <= NC; j++) {
            double ue = 0.5 * (u[i][j] + u[i + 1][j]), uw = 0.5 * (u[i - 1][j] + u[i][j]);
            double un = 0.5 * (u[i][j] + u[i][j + 1]), us = 0.5 * (u[i][j - 1] + u[i][j]);
            double vn = 0.5 * (v[i][j] + v[i + 1][j]), vs = 0.5 * (v[i][j - 1] + v[i + 1][j - 1]);
            double conv = (ue * ue - uw * uw) / DX + (un * vn - us * vs) / DY;
            double diff = (u[i + 1][j] - 2 * u[i][j] + u[i - 1][j]) / (DX * DX)
                        + (u[i][j + 1] - 2 * u[i][j] + u[i][j - 1]) / (DY * DY);
            unew[i][j] = u[i][j] + DT * (-conv + diff / RE - (p[i + 1][j] - p[i][j]) / DX);
        }
    }
    for (i = 1; i <= NC; i++) {
        for (j = 1; j < NC; j++) {
            double vn = 0.5 * (v[i][j] + v[i][j + 1]), vs = 0.5 * (v[i][j - 1] + v[i][j]);
            double ve = 0.5 * (v[i][j] + v[i + 1][j]), vw = 0.5 * (v[i - 1][j] + v[i][j]);
            double ue = 0.5 * (u[i][j] + u[i][j + 1]), uw = 0.5 * (u[i - 1][j] + u[i - 1][j + 1]);
            double conv = (ue * ve - uw * vw) / DX + (vn * vn - vs * vs) / DY;
            double diff = (v[i + 1][j] - 2 * v[i][j] + v[i - 1][j]) / (DX * DX)
                        + (v[i][j + 1] - 2 * v[i][j] + v[i][j - 1]) / (DY * DY);
            vnew[i][j] = v[i][j] + DT * (-conv + diff / RE - (p[i][j + 1] - p[i][j]) / DY);
        }
    }
    apply_velocity_bc(unew, vnew);
}

// [COMPLETED] (b) corrector: project u* onto a divergence-free field, update pressure.
// Returns max |u^{n+1} - u^n| for the steady-state test. [FIX: original never copied unew -> u]
double update_velocity() {
    int i, j; double change = 0.0;
    for (i = 1; i < NC; i++) for (j = 1; j <= NC; j++) {
        double un1 = unew[i][j] - DT * (pnew[i + 1][j] - pnew[i][j]) / DX;
        if (fabs(un1 - u[i][j]) > change) change = fabs(un1 - u[i][j]);
        u[i][j] = un1;
    }
    for (i = 1; i <= NC; i++) for (j = 1; j < NC; j++) {
        double vn1 = vnew[i][j] - DT * (pnew[i][j + 1] - pnew[i][j]) / DY;
        if (fabs(vn1 - v[i][j]) > change) change = fabs(vn1 - v[i][j]);
        v[i][j] = vn1;
    }
    for (i = 1; i <= NC; i++) for (j = 1; j <= NC; j++) p[i][j] += pnew[i][j];
    apply_velocity_bc(u, v);
    return change;
}

// Function to compute vorticity
// [COMPLETED] vorticity at cell corners: w = dv/dx - du/dy; stream function from integrating
// u = dpsi/dy along y (psi = 0 on all walls)
void compute_vorticity() {
    int i, j;
    for (i = 0; i <= NC; i++) {
        for (j = 0; j <= NC; j++) {
            Vor[i][j] = (v[i + 1][j] - v[i][j]) / DX - (u[i][j + 1] - u[i][j]) / DY;
        }
        S[i][0] = 0.0;
        for (j = 1; j <= NC; j++) S[i][j] = S[i][j - 1] + u[i][j] * DY;
    }
}

int main() {
    int i, j, t;
    FILE *streamlineFile, *vorticityFile, *pressureFile, *uFile, *vFile, *histFile;

    // Open files for writing
    streamlineFile = fopen("streamlines.txt", "w");
    vorticityFile = fopen("vorticity.txt", "w");
    pressureFile = fopen("pressure.txt", "w");
    uFile = fopen("u_values.txt", "w");
    vFile = fopen("v_values.txt", "w");
    histFile = fopen("convergence_history.txt", "w");   // [ADDED]

    if (streamlineFile == NULL || vorticityFile == NULL || pressureFile == NULL || uFile == NULL || vFile == NULL) {
        printf("Error opening files.\n");
        return 1; // Exit the program if the file opening fails
    }

    // Initialize the velocity and pressure fields
    for (i = 0; i <= NC; i++) for (j = 0; j <= NC + 1; j++) u[i][j] = 0.0;
    for (i = 0; i <= NC + 1; i++) for (j = 0; j <= NC; j++) v[i][j] = 0.0;
    for (i = 0; i <= NC + 1; i++) for (j = 0; j <= NC + 1; j++) p[i][j] = 0.0;
    apply_velocity_bc(u, v);
    printf("SMAC lid-driven cavity: Re=%.0f, %dx%d cells, dt=%g, CFL=%.3f, diffusion number=%.3f\n",
           RE, NC, NC, DT, U_INFTY * DT / DX, DT / (RE * DX * DX));

    // Time-stepping loop
    double change = 1.0;
    for (t = 1; t <= NT; t++) {
        predict_velocity();
        int sor_it = compute_pressure();
        change = update_velocity();
        if (t % 100 == 0) {
            fprintf(histFile, "%d %.6f %.6e %d\n", t, t * DT, change, sor_it);
            if (t % 2000 == 0) printf("step %6d  t=%7.3f  max|du|=%.3e  SOR its=%d\n", t, t * DT, change, sor_it);
        }
        if (change < STEADY_TOL) { printf("Steady state reached at step %d (t = %.3f)\n", t, t * DT); break; }
    }
    compute_vorticity();

    /// Write values to files  [FIX: values interpolated to the (NC+1)x(NC+1) cell-corner nodes;
    ///  columns: x y value, one block per y-row]
    for (j = 0; j <= NC; j++) {
        for (i = 0; i <= NC; i++) {
            double x = i * DX, y = j * DY;
            double uc = 0.5 * (u[i][j] + u[i][j + 1]);
            double vc = 0.5 * (v[i][j] + v[i + 1][j]);
            double pc = 0.25 * (p[i][j] + p[i + 1][j] + p[i][j + 1] + p[i + 1][j + 1]);
            fprintf(streamlineFile, "%f %f %e\n", x, y, S[i][j]);
            fprintf(vorticityFile, "%f %f %e\n", x, y, Vor[i][j]);
            fprintf(pressureFile, "%f %f %e\n", x, y, pc);
            fprintf(uFile, "%f %f %e\n", x, y, uc);
            fprintf(vFile, "%f %f %e\n", x, y, vc);
        }
    }

    // Close files
    fclose(streamlineFile);
    fclose(vorticityFile);
    fclose(pressureFile);
    fclose(uFile);
    fclose(vFile);
    fclose(histFile);

    return 0;
}
