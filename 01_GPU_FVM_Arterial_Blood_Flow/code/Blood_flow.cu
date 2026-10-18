// Blood through arteries (LAD) artery analysis
// The flow is assumed to be unsteady, incompressible, laminar, non-Newtonian, and fully developed
//
// Original author: Karthick J (OE23S018), OE5450 Numerical Techniques in Ocean Hydrodynamics, final project.
// Revision: corrections are marked [FIX], additions [ADDED], behaviour changes [CHANGED]. The kernel
// structure of the original (Initialize, ApplyBoundaryConditions, Shear_rate, X/YMomentum_solver,
// Divergence, PoissonSolverGS, CorrectVelocities, CalculateResidual, CalculateMaxVelocity, writeVTKFile)
// is kept. Method: explicit fractional-step (projection) method on a MAC staggered grid, QUICK
// convection, central diffusion with Bird-Carreau viscosity, red-black SOR pressure correction.
//
// Build (GPU):  nvcc -O3 -arch=native -o blood_flow Blood_flow.cu
// Run:          ./blood_flow [physical_time_s] [vtk_interval_steps] [sor_omega] [max_sor_sweeps] [block_correction 0/1]
//               defaults:     5.0               1000                 1.85        1000             1

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Defining the variables for the 2D flow simulation for host and device
#define rho 1.06e3f   // density of blood in kg/m^3
#define L  0.14f      // length of the artery in m
#define W  0.0035f    // width of the artery in m
#define U  0.39f      // velocity of the blood in m/s in x direction in the artery
                      // [FIX: U is the time-averaged MEAN velocity of the report (LAD data, 0.39 m/s);
                      //  the inlet profile below is scaled so that its mean is U]
#define V  0.0f       // velocity of the blood in m/s in y direction in the artery
#define DX 0.0001f    // length of the cell in m for x direction (100micrometers)
#define DY 0.0001f    // length of the cell in m for y direction (100micrometers)
#define dt 1.0e-5f    // time step in seconds [FIX: no longer used inside the kernels, which now receive
                      //  the adaptive time step actually taken (see main)]
#define NX ((int)(L/DX + 0.5f))   // number of cells in x direction  [FIX: was (int)(L/DX)+1. The arrays
                                  //  hold CELLS, so +1 (a node count) gave 1401 x 36 cells = 140.1 x 3.6 mm;
                                  //  rounding instead of truncating also protects against L/DX = 1399.99..]
#define NY ((int)(W/DY + 0.5f))   // number of cells in y direction  [FIX: 35 cells across W = 3.5 mm]
#define N  (NX*NY)           // total number of cells
#define del_P 2.8863e3f      // initial pressure difference in Pa using the Poiseuille equation

// Bird-Carreau model parameters
#define mu0    0.056f   // dynamic viscosity of blood in Pa.s at zero
#define mu_inf 0.0035f  // dynamic viscosity of blood in Pa.s at infinite
#define lambda 3.13f    // time constant in seconds
#define n      0.6f     // flow index for shear thinning fluid
#define a      2.0f     // bird-carreau model parameter

// Forward declarations
// [FIX: the original prototypes did not compile: parameters named dt, DX, DY, rho collide with the
//  macros above, and two prototypes contained "int 1401, int 36". Parameters renamed dt_val, dx, dy,
//  rho_val; nx, ny restored.]
__global__ void Initialize(float *u, float *v, float *p, float *visc, float *shear_rate, float p_init, int nx, int ny);
__global__ void ApplyBoundaryConditions(float *u, float *v, float *p, float p_out, int nx, int ny);
__global__ void Shear_rate(float *u, float *v, float *mu, float *shear_rate, int nx, int ny);
__global__ void Xmomentum_solver(float *u_x, float *u_y, float *p, float *mu, float *u_x_new, float dt_val, float dx, float dy, float rho_val, int nx, int ny);
__global__ void YMomentum_solver(float* u_x, float* u_y, float* p, float* v_new, float dt_val, float dx, float dy, float rho_val, float* mu, int nx, int ny);
__global__ void Divergence(float *u_x_new, float *v_new, float *divergence, int nx, int ny);
__global__ void PoissonSolverGS(float *divergence, float *p_prime, int nx, int ny, float omega, float dt_val, int color, float *residual);
__global__ void ColumnResidual(float *divergence, float *p_prime, int nx, int ny, float dt_val, float *col_res);
__global__ void BlockCorrectionTDMA(float *col_res, float *col_corr, float *work, int nx, int ny);
__global__ void ApplyBlockCorrection(float *p_prime, float *col_corr, int nx, int ny);
__global__ void CorrectVelocities(float *u, float *v, float *u_star, float *v_star, float *p, float *p_prime, float dt_val, int nx, int ny);
__global__ void CalculateResidual(float *d_u, float *d_u_old, float *d_v, float *d_v_old, float *residual, int nx, int ny);
__global__ void CalculateMaxVelocity(float *d_u, float *d_v, float *d_mu, float *max_velocity, float *max_u2_mu, float *max_inv_dt, int nx, int ny);
bool SolvePressurePoisson(float *d_divergence, float *d_p_prime, int nx, int ny, float omega, float tolerance,
                          float dt_val, int max_iter, int block_corr, float *d_residual, float *d_col_res,
                          float *d_col_corr, float *d_work, int *iters_done);
void writeVTKFile(const char* filename, float *u, float *v, float *p, float *visc, float *shear_rate, int nx, int ny, float dx, float dy);

// Bird-Carreau viscosity  mu = mu_inf + (mu0 - mu_inf) [1 + (lambda*gamma)^a]^((n-1)/a)
// [ADDED as a function so that the kernels and the host-side wall-shear-stress output use one definition]
__host__ __device__ inline float carreau(float sr) {
    return mu_inf + (mu0 - mu_inf) * powf(1.0f + powf(lambda * sr, a), (n-1.0f)/a);
}

// QUICK face value: phi_f = 6/8 phi_U + 3/8 phi_D - 1/8 phi_UU (U = upstream, D = downstream,
// UU = far upstream). [ADDED helper; same formula as the original. FIX: falls back to first-order
// upwind (phi_f = phi_U) when UU lies outside the domain -- the original read u(i-2), u(j-2), u(j+2)
// at the first/last interior faces, i.e. outside the arrays.]
__device__ inline float quick(float phi_D, float phi_U, float phi_UU, bool has_UU) {
    return has_UU ? (0.375f*phi_D + 0.75f*phi_U - 0.125f*phi_UU) : phi_U;
}

// Initialize flow field with initial conditions
__global__ void Initialize(float *u, float *v, float *p, float *visc, float *shear_rate, float p_init, int nx, int ny) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    // Initialize pressure and viscosity at cell centers
    if (i < nx && j < ny) {
        int idx = i + j*nx;  // Cell-centered index
        p[idx] = p_init * (1.0f - (float)i/nx);  // Linear pressure gradient
        visc[idx] = mu0;  // Initialize with zero-shear viscosity
        shear_rate[idx] = 0.0f;  // Initialize shear rate to zero
    }

    // Initialize x-velocity at east/west faces (staggered grid)
    if (i <= nx && j < ny) {  // Note: i can be nx for the last face
        int idx_u = i + j*(nx+1);  // Face-centered index for u

        if (i == 0) {
             u[idx_u] = U;
        } else if (i == nx) {
            // Outlet boundary condition
            u[idx_u] = 0.0f;  // [FIX: was u[idx_u-1], read while another thread initialises it (race)]
        } else {
            // Interior points
            u[idx_u] = 0.0f;  // Initialize to zero
        }
    }

    // Initialize y-velocity at north/south faces (staggered grid)
    if (i < nx && j <= ny) {  // Note: j can be ny for the last face
        int idx_v = i + j*nx;  // Face-centered index for v
        v[idx_v] = 0.0f;  // Initialize to zero

        // No-slip boundary conditions at walls (north/south)
        if (j == 0 || j == ny) {
            v[idx_v] = 0.0f;
        }
    }
}

// Apply boundary conditions for staggered grid
// Layout: p, mu at cell centres (i, j); u at the west face of cell i, u[i + j*(nx+1)], i = 0..nx;
//         v at the south face of cell j, v[i + j*nx], j = 0..ny.
__global__ void ApplyBoundaryConditions(float *u, float *v, float *p, float p_out, int nx, int ny) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    // Inlet boundary (west face, x = 0)
    if (i == 0 && j < ny) {
        // u-velocity at inlet face (i=0)
        // [FIX: u in row j sits at y = (j+0.5)DY, so the walls are at eta = -1, +1 only for
        //  eta = 2(j+0.5)/ny - 1 (the original 2j/(ny-1) - 1 put u = 0 in the first and last fluid
        //  rows), and the factor 1.5 makes the mean inflow equal to U (U(1-eta^2) has mean 2U/3)]
        float y_norm = 2.0f * ((float)j + 0.5f)/ny - 1.0f;
        u[j*(nx+1) + i] = 1.5f * U * (1.0f - y_norm*y_norm);  // Parabolic profile

        // v-velocity near inlet
        v[j*nx + i] = 0.0f;  // No y-component at inlet

        // [FIX: removed p[inlet cell] = p[next cell]. Equal pressures on both sides of face 1 removed the
        //  pressure force on that face; the inlet pressure condition (zero normal gradient of the
        //  pressure correction, consistent with a prescribed inlet velocity) is applied in PoissonSolverGS]
    }

    // Outlet boundary (east face, x = nx)
    if (i == nx && j < ny) {
        // u-velocity at outlet face (i=nx)
        u[j*(nx+1) + i] = u[j*(nx+1) + (i-1)];  // Zero gradient

        // Pressure at outlet cells (i=nx-1)
        if (i-1 >= 0) {
            p[j*nx + (i-1)] = p_out;  // Fixed pressure at outlet
        }
    }

    // Wall boundaries (south face, y = 0)
    if (j == 0 && i < nx) {
        // v-velocity at bottom wall (j=0)
        v[j*nx + i] = 0.0f;  // No-slip condition
        // [FIX: removed u(row 0) = 0. On the staggered grid that row is at y = DY/2, inside the fluid;
        //  forcing it to zero moved the wall one cell inwards (33 instead of 35 fluid rows). No-slip is
        //  imposed with the ghost value u(-1) = -u(0) in the momentum and shear-rate kernels.]
    }

    // Wall boundaries (north face, y = ny)
    if (j == ny && i < nx) {
        // v-velocity at top wall (j=ny)
        v[j*nx + i] = 0.0f;  // No-slip condition
        // [FIX: removed u(row ny-1) = 0 for the same reason (ghost value u(ny) = -u(ny-1))]
    }
}

// Calculate shear rate and update viscosity using Bird-Carreau model
__global__ void Shear_rate(float *u, float *v, float *mu, float *shear_rate, int nx, int ny) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    // [FIX: evaluated in EVERY cell. The original skipped the wall-adjacent and inlet/outlet cells and
    //  set mu = mu0 there, i.e. the highest viscosity exactly where the shear rate is largest.]
    if (i < nx && j < ny) {
        // Normal strain rates at the cell centre (exact on the staggered grid)
        float du_dx = (u[j*(nx+1) + (i+1)] - u[j*(nx+1) + i]) / DX;
        float dv_dy = (v[(j+1)*nx + i] - v[j*nx + i]) / DY;

        // [FIX: cross derivatives by central differences of cell-centred velocities. The original
        //  (u_{j+1} - u_j)/DY (two-face average) is du/dy at y_{j+1/2}, half a cell above the centre,
        //  and the same for dv/dx.] Ghost values: walls u(-1) = -u(0), u(ny) = -u(ny-1) (no slip);
        //  inlet v = 0 (antisymmetric ghost); outlet zero gradient.
        float uc  = 0.5f * (u[j*(nx+1) + i] + u[j*(nx+1) + (i+1)]);
        float ucN = (j < ny-1) ? 0.5f * (u[(j+1)*(nx+1) + i] + u[(j+1)*(nx+1) + (i+1)]) : -uc;
        float ucS = (j > 0)    ? 0.5f * (u[(j-1)*(nx+1) + i] + u[(j-1)*(nx+1) + (i+1)]) : -uc;
        float vc  = 0.5f * (v[j*nx + i] + v[(j+1)*nx + i]);
        float vcE = (i < nx-1) ? 0.5f * (v[j*nx + (i+1)] + v[(j+1)*nx + (i+1)]) : vc;
        float vcW = (i > 0)    ? 0.5f * (v[j*nx + (i-1)] + v[(j+1)*nx + (i-1)]) : -vc;
        float du_dy = (ucN - ucS) / (2.0f * DY);
        float dv_dx = (vcE - vcW) / (2.0f * DX);

        // Calculate shear rate
        float sr = sqrtf(2.0f * (du_dx*du_dx + dv_dy*dv_dy) + (du_dy + dv_dx)*(du_dy + dv_dx));

        // Store at cell center
        int idx = j*nx + i;
        shear_rate[idx] = sr;

        // Update viscosity using Bird-Carreau model
        mu[idx] = carreau(sr);
    }
}

// X-momentum equation solver (explicit Euler predictor u*)
__global__ void Xmomentum_solver(float *u_x, float *u_y, float *p, float *mu, float *u_x_new, float dt_val, float dx, float dy, float rho_val, int nx, int ny) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i > nx || j >= ny) return;
    int idx_u = j*(nx+1) + i;

    // [FIX: the inlet (i = 0) and outlet (i = nx) faces are copied so that u* exists on every face;
    //  the original left them uninitialised (cudaMalloc memory) and Divergence/CorrectVelocities read them]
    if (i == 0 || i == nx) { u_x_new[idx_u] = u_x[idx_u]; return; }

    // [FIX: solved on faces i = 1..nx-1 and ALL rows j = 0..ny-1. The original range i < nx-1,
    //  1 <= j < ny-1 never updated face nx-1 (so the outlet face stayed at u = 0, blocking the
    //  channel) and froze the two wall-adjacent rows.]
    float uP = u_x[idx_u];
    float uE = u_x[idx_u + 1];
    float uW = u_x[idx_u - 1];
    float uN = (j < ny-1) ? u_x[idx_u + (nx+1)] : -uP;   // ghost value: no-slip wall at y = W
    float uS = (j > 0)    ? u_x[idx_u - (nx+1)] : -uP;   // ghost value: no-slip wall at y = 0

    // 1. Calculate convection term using QUICK scheme
    // Calculate mass fluxes through cell faces (u-control volume centred on face i, between cells i-1 and i)
    float Fe = 0.5f * (uP + uE) * dy;
    float Fw = 0.5f * (uW + uP) * dy;

    // Interpolate v-velocities to u-cell faces
    // [FIX: two-point averages at the north and south faces of the u-CV. The original four-point
    //  average is v at the CV centre (x_i, y_j) and was used for both faces.]
    float vn = 0.5f * (u_y[(j+1)*nx + i] + u_y[(j+1)*nx + (i-1)]);
    float vs = 0.5f * (u_y[j*nx + i]     + u_y[j*nx + (i-1)]);

    float Fn = vn * dx;
    float Fs = vs * dx;

    // QUICK scheme for face values
    float phi_e, phi_w, phi_n, phi_s;
    if (Fe >= 0) phi_e = quick(uE, uP, uW, true);
    else         phi_e = quick(uP, uE, (i+2 <= nx) ? u_x[idx_u + 2] : 0.0f, i+2 <= nx);
    if (Fw >= 0) phi_w = quick(uP, uW, (i-2 >= 0) ? u_x[idx_u - 2] : 0.0f, i-2 >= 0);
    else         phi_w = quick(uW, uP, uE, true);
    if (Fn >= 0) phi_n = quick(uN, uP, uS, j-1 >= 0);
    else         phi_n = quick(uP, uN, (j+2 <= ny-1) ? u_x[idx_u + 2*(nx+1)] : 0.0f, j+2 <= ny-1);
    if (Fs >= 0) phi_s = quick(uP, uS, (j-2 >= 0) ? u_x[idx_u - 2*(nx+1)] : 0.0f, j-2 >= 0);
    else         phi_s = quick(uS, uP, uN, j+1 <= ny-1);

    float convection_term = -(Fe * phi_e - Fw * phi_w + Fn * phi_n - Fs * phi_s) / (dx * dy);

    // 2. Calculate diffusion term using central difference
    // [FIX: viscosity at the faces of the u-CV. East/west faces are the cell centres i and i-1; north/
    //  south faces are cell corners (4-cell average, 2 cells at a wall). The original averaged
    //  mu(i,j) with mu(i+1,j) and mu(i,j+1), i.e. values shifted by half a cell.]
    float mu_e = mu[j*nx + i];
    float mu_w = mu[j*nx + (i-1)];
    float mu_n = (j < ny-1) ? 0.25f * (mu[j*nx + i] + mu[j*nx + (i-1)] + mu[(j+1)*nx + i] + mu[(j+1)*nx + (i-1)])
                            : 0.5f  * (mu[j*nx + i] + mu[j*nx + (i-1)]);
    float mu_s = (j > 0)    ? 0.25f * (mu[j*nx + i] + mu[j*nx + (i-1)] + mu[(j-1)*nx + i] + mu[(j-1)*nx + (i-1)])
                            : 0.5f  * (mu[j*nx + i] + mu[j*nx + (i-1)]);

    // Calculate diffusion coefficients
    float De = mu_e * dy/dx;
    float Dw = mu_w * dy/dx;
    float Dn = mu_n * dx/dy;
    float Ds = mu_s * dx/dy;

    // Calculate diffusion term using coefficients
    float diffusion = De * (uE - uP) - Dw * (uP - uW) + Dn * (uN - uP) - Ds * (uP - uS);

    // 3. Calculate pressure gradient (staggered grid - pressure at cell centers)
    float pressure_gradient_term = (p[j*nx + i] - p[j*nx + (i-1)]) / dx;

    // 4. Update u_x using the momentum equation
    u_x_new[idx_u] = uP + dt_val * (convection_term + diffusion/(dx*dy*rho_val) - pressure_gradient_term/rho_val);
}

// Y-momentum equation solver (explicit Euler predictor v*)
__global__ void YMomentum_solver(float* u_x, float* u_y, float* p, float* v_new, float dt_val, float dx, float dy, float rho_val, float* mu, int nx, int ny) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= nx || j > ny) return;
    int idx = i + j*nx;

    // Faces that are not solved keep their boundary value: walls (j = 0, ny) and, as in the original,
    // the inlet and outlet cell columns (i = 0, nx-1) where v = 0 (developed in/outflow).
    // [FIX: they are copied into v* (the original left them uninitialised)]
    if (i < 1 || i > nx-2 || j < 1 || j > ny-1) { v_new[idx] = u_y[idx]; return; }
    // [FIX: solved for j = 1..ny-1; the original j < ny-1 froze the interior face j = ny-1 at v = 0]

    float vP = u_y[idx];
    float vE = u_y[idx + 1];
    float vW = u_y[idx - 1];
    float vN = u_y[idx + nx];
    float vS = u_y[idx - nx];

    // Interpolate u-velocities to v-cell faces (east and west)
    // [FIX: two-point averages at the east face (u at x_{i+1}, rows j-1 and j) and west face (x_i);
    //  the original four-point average is u at the CV centre and was used for both faces]
    float ue = 0.5f * (u_x[(i+1) + j*(nx+1)] + u_x[(i+1) + (j-1)*(nx+1)]);
    float Fe = ue * dy;
    float uw = 0.5f * (u_x[i + j*(nx+1)] + u_x[i + (j-1)*(nx+1)]);
    float Fw = uw * dy;

    // Interpolate v-velocities to v-cell faces (north and south)
    float vn = 0.5f * (vP + vN);
    float Fn = vn * dx;
    float vs = 0.5f * (vS + vP);
    float Fs = vs * dx;

    // QUICK scheme for face values with flow direction check
    float phi_e, phi_w, phi_n, phi_s;
    if (Fe >= 0) phi_e = quick(vE, vP, vW, true);
    else         phi_e = quick(vP, vE, (i+2 <= nx-1) ? u_y[idx + 2] : 0.0f, i+2 <= nx-1);
    if (Fw >= 0) phi_w = quick(vP, vW, (i-2 >= 0) ? u_y[idx - 2] : 0.0f, i-2 >= 0);
    else         phi_w = quick(vW, vP, vE, true);
    if (Fn >= 0) phi_n = quick(vN, vP, vS, true);
    else         phi_n = quick(vP, vN, (j+2 <= ny) ? u_y[idx + 2*nx] : 0.0f, j+2 <= ny);
    if (Fs >= 0) phi_s = quick(vP, vS, (j-2 >= 0) ? u_y[idx - 2*nx] : 0.0f, j-2 >= 0);
    else         phi_s = quick(vS, vP, vN, true);

    // [FIX: viscosity at the faces of the v-CV: north/south faces are the cell centres (i,j) and
    //  (i,j-1); east/west faces are cell corners (4-cell averages)]
    float mu_n = mu[i + j*nx];
    float mu_s = mu[i + (j-1)*nx];
    float mu_e = 0.25f * (mu[i + j*nx] + mu[(i+1) + j*nx] + mu[i + (j-1)*nx] + mu[(i+1) + (j-1)*nx]);
    float mu_w = 0.25f * (mu[i + j*nx] + mu[(i-1) + j*nx] + mu[i + (j-1)*nx] + mu[(i-1) + (j-1)*nx]);

    // Calculate diffusion coefficients
    float De = mu_e * dy/dx;
    float Dw = mu_w * dy/dx;
    float Dn = mu_n * dx/dy;
    float Ds = mu_s * dx/dy;

    // Calculate convection term
    float convection_term = -(Fe * phi_e - Fw * phi_w + Fn * phi_n - Fs * phi_s) / (dx * dy);

    // Calculate diffusion term using coefficients
    float diffusion = De * (vE - vP) - Dw * (vP - vW) + Dn * (vN - vP) - Ds * (vP - vS);

    // Pressure gradient for v-velocity at (i,j-1/2)
    float pressure_gradient_term = (p[i + j*nx] - p[i + (j-1)*nx]) / dy;

    // Update v-velocity
    v_new[idx] = vP + dt_val * (convection_term + diffusion/(dx*dy*rho_val) - pressure_gradient_term/rho_val);
}

// Calculate divergence for pressure correction
__global__ void Divergence(float *u_x_new, float *v_new, float *divergence, int nx, int ny) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    // [FIX: every cell (the original skipped the boundary cells, whose divergence stayed uninitialised
    //  and whose mass balance was never enforced)]
    if (i < nx && j < ny) {
        float du_dx = (u_x_new[j*(nx+1) + (i+1)] - u_x_new[j*(nx+1) + i]) / DX;
        float dv_dy = (v_new[(j+1)*nx + i] - v_new[j*nx + i]) / DY;
        divergence[j*nx + i] = du_dx + dv_dy;
    }
}

// Pressure-correction equation  lap(p') = (rho/dt) div(u*)  by red-black Gauss-Seidel with SOR.
// Unknowns: cells i = 0..nx-2 (the last column holds the outlet pressure, p' = 0). A neighbour
// coefficient is zero where the corresponding face velocity is not corrected: inlet face (u prescribed),
// walls (v = 0) and the v-faces of the inlet column (v = 0, see YMomentum_solver).
__global__ void PoissonSolverGS(float *divergence, float *p_prime, int nx, int ny, float omega, float dt_val, int color, float *residual) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    // [FIX: red-black ordering. The original updated every cell in one launch while reading neighbours
    //  from the array being written (race: a non-deterministic mix of Jacobi and Gauss-Seidel), and it
    //  never updated the boundary cells, so no pressure boundary condition entered the solve.]
    if (i >= nx-1 || j >= ny || ((i + j) & 1) != color) return;
    int idx = j*nx + i;

    float aE = 1.0f/(DX*DX);
    float aW = (i > 0)              ? 1.0f/(DX*DX) : 0.0f;
    float aN = (i > 0 && j < ny-1)  ? 1.0f/(DY*DY) : 0.0f;
    float aS = (i > 0 && j > 0)     ? 1.0f/(DY*DY) : 0.0f;
    float sum = aE * p_prime[idx + 1];
    if (aW > 0.0f) sum += aW * p_prime[idx - 1];
    if (aN > 0.0f) sum += aN * p_prime[idx + nx];
    if (aS > 0.0f) sum += aS * p_prime[idx - nx];

    // [FIX: source sign and time step. u = u* - (dt/rho) grad p' with div u = 0 requires
    //  lap p' = +(rho/dt) div u*; the original used -rho*div/dt with the constant macro dt = 1e-5 s
    //  instead of the adaptive time step actually taken.]
    float p_old = p_prime[idx];
    float p_gs = (sum - rho * divergence[idx] / dt_val) / (aE + aW + aN + aS);

    // Apply SOR (Successive Over-Relaxation)
    float p_new = (1.0f - omega) * p_old + omega * p_gs;
    p_prime[idx] = p_new;

    // Calculate local residual for convergence check
    atomicAdd(residual, fabsf(p_new - p_old));
}

// [ADDED] Block (additive) correction along the artery (Settari & Aziz 1973; Patankar 1980, Sec. 7.2).
// Point SOR on a 1400 x 35 domain reduces the error component that is uniform across the artery
// (a change of the axial flow rate) by only ~1e-5 per sweep, so the original solver could not make
// the cross-sectional flow rate equal to the inflow. The correction adds c_i to every cell of column i,
// with c_i chosen so that the column-summed residual vanishes (exact mass balance of every cross-section).
__global__ void ColumnResidual(float *divergence, float *p_prime, int nx, int ny, float dt_val, float *col_res) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nx-1) return;
    float R = 0.0f;
    for (int j = 0; j < ny; j++) {
        int idx = j*nx + i;
        float pP = p_prime[idx];
        float r = rho * divergence[idx] / dt_val - (p_prime[idx + 1] - pP) / (DX*DX);
        if (i > 0) {
            r -= (p_prime[idx - 1] - pP) / (DX*DX);
            if (j < ny-1) r -= (p_prime[idx + nx] - pP) / (DY*DY);
            if (j > 0)    r -= (p_prime[idx - nx] - pP) / (DY*DY);
        }
        R += r;   // residual of  sum_nb a_nb (p'_nb - p'_P) = (rho/dt) div
    }
    col_res[i] = R;
}

// Single-thread Thomas algorithm for  A_W c_{i-1} - (A_W + A_E) c_i + A_E c_{i+1} = R_i,
// i = 0..nx-2, A_E = A_W = ny/DX^2 (A_W = 0 at the inlet), c_{nx-1} = 0 (outlet column).
__global__ void BlockCorrectionTDMA(float *col_res, float *col_corr, float *work, int nx, int ny) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    int m = nx - 1;
    float A = (float)ny / (DX*DX);
    // forward elimination (work = modified super-diagonal)
    float b0 = -A;                              // i = 0: A_W = 0
    work[0] = A / b0;
    col_corr[0] = col_res[0] / b0;
    for (int i = 1; i < m; i++) {
        float bi = -2.0f*A;
        float den = bi - A * work[i-1];
        work[i] = (i < m-1) ? A / den : 0.0f;   // c_{nx-1} = 0
        col_corr[i] = (col_res[i] - A * col_corr[i-1]) / den;
    }
    for (int i = m-2; i >= 0; i--) col_corr[i] -= work[i] * col_corr[i+1];
}

__global__ void ApplyBlockCorrection(float *p_prime, float *col_corr, int nx, int ny) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < nx-1 && j < ny) p_prime[j*nx + i] += col_corr[i];
}

// Host-side wrapper for the red-black SOR iteration with convergence check
bool SolvePressurePoisson(float *d_divergence, float *d_p_prime, int nx, int ny, float omega, float tolerance,
                          float dt_val, int max_iter, int block_corr, float *d_residual, float *d_col_res,
                          float *d_col_corr, float *d_work, int *iters_done) {
    dim3 block(16, 16);
    dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);
    dim3 block1(128);
    dim3 grid1((nx + block1.x - 1) / block1.x);
    float h_residual;

    // [FIX: incremental formulation. The momentum predictor already contains grad p^n, so the Poisson
    //  equation is solved for the correction p' (zero initial guess, i.e. warm start p^{n+1} = p^n);
    //  the original solved for the full pressure and then applied grad p a second time.]
    cudaMemset(d_p_prime, 0, nx * ny * sizeof(float));

    for (int iter = 0; iter < max_iter; ++iter) {
        if (block_corr && iter % 10 == 0) {   // [ADDED] block correction every 10 sweeps
            ColumnResidual<<<grid1, block1>>>(d_divergence, d_p_prime, nx, ny, dt_val, d_col_res);
            BlockCorrectionTDMA<<<1, 1>>>(d_col_res, d_col_corr, d_work, nx, ny);
            ApplyBlockCorrection<<<grid, block>>>(d_p_prime, d_col_corr, nx, ny);
        }
        // Reset residual
        cudaMemset(d_residual, 0, sizeof(float));

        // Perform one red-black SOR sweep (two half-sweeps)
        PoissonSolverGS<<<grid, block>>>(d_divergence, d_p_prime, nx, ny, omega, dt_val, 0, d_residual);
        PoissonSolverGS<<<grid, block>>>(d_divergence, d_p_prime, nx, ny, omega, dt_val, 1, d_residual);

        // Check convergence (mean |change| per sweep)
        cudaMemcpy(&h_residual, d_residual, sizeof(float), cudaMemcpyDeviceToHost);
        h_residual /= (nx * ny);  // Normalize by grid size

        if (h_residual < tolerance) {
            *iters_done = iter + 1;
            if (block_corr) {
                ColumnResidual<<<grid1, block1>>>(d_divergence, d_p_prime, nx, ny, dt_val, d_col_res);
                BlockCorrectionTDMA<<<1, 1>>>(d_col_res, d_col_corr, d_work, nx, ny);
                ApplyBlockCorrection<<<grid, block>>>(d_p_prime, d_col_corr, nx, ny);
            }
            return true;  // Converged
        }
    }
    *iters_done = max_iter;
    if (block_corr) {   // [ADDED] final block correction: exact column mass balance on exit
        ColumnResidual<<<grid1, block1>>>(d_divergence, d_p_prime, nx, ny, dt_val, d_col_res);
        BlockCorrectionTDMA<<<1, 1>>>(d_col_res, d_col_corr, d_work, nx, ny);
        ApplyBlockCorrection<<<grid, block>>>(d_p_prime, d_col_corr, nx, ny);
    }
    return false;  // Did not converge within max_iter
}

// Correct velocities using the pressure correction, and update the pressure
__global__ void CorrectVelocities(float *u, float *v, float *u_star, float *v_star, float *p, float *p_prime, float dt_val, int nx, int ny) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    // d-coefficient of the projection step [FIX: adaptive dt (was the constant macro dt)]
    float d_c = dt_val / rho;

    // Correct u-velocity (staggered, west face of cell i)
    // [FIX: all interior faces i = 1..nx-1 and all rows (was 1 <= j < ny-1); boundary faces take u*]
    if (i <= nx && j < ny) {
        int idx_u = j*(nx+1) + i;
        if (i >= 1 && i <= nx-1) {
            float dp_dx = (p_prime[j*nx + i] - p_prime[j*nx + (i-1)]) / DX;  // [FIX: gradient of p', not p]
            u[idx_u] = u_star[idx_u] - d_c * dp_dx;
        } else {
            u[idx_u] = u_star[idx_u];
        }
    }

    // Correct v-velocity (staggered, south face of cell j) on the faces solved in YMomentum_solver
    if (i < nx && j <= ny) {
        int idx_v = j*nx + i;
        if (i >= 1 && i <= nx-2 && j >= 1 && j <= ny-1) {
            float dp_dy = (p_prime[j*nx + i] - p_prime[(j-1)*nx + i]) / DY;
            v[idx_v] = v_star[idx_v] - d_c * dp_dy;
        } else {
            v[idx_v] = v_star[idx_v];
        }
    }

    // [ADDED] pressure update p^{n+1} = p^n + p' (the original never updated p from the correction)
    if (i < nx && j < ny) p[j*nx + i] += p_prime[j*nx + i];
}

// Calculate residual between time steps (sum of squared changes)
__global__ void CalculateResidual(float *d_u, float *d_u_old, float *d_v, float *d_v_old, float *residual, int nx, int ny) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    // [FIX: all (nx+1) x ny u-faces and nx x (ny+1) v-faces. Inside "if (i < nx && j < ny)" the tests
    //  i < nx+1 and j < ny+1 were always true, so the outlet faces and the top-wall row were skipped.]
    if (i <= nx && j < ny) {
        float u_diff = d_u[j*(nx+1) + i] - d_u_old[j*(nx+1) + i];
        atomicAdd(residual, u_diff*u_diff);
    }
    if (i < nx && j <= ny) {
        float v_diff = d_v[j*nx + i] - d_v_old[j*nx + i];
        atomicAdd(residual, v_diff*v_diff);
    }
}

// Calculate maximum velocity (CFL), and the two explicit-stability indicators [ADDED]:
//   max_inv_dt = max over cells of |u|/DX + |v|/DY + 2(mu/rho)(1/DX^2 + 1/DY^2)   (high-wavenumber bound)
//   max_u2_mu  = max over cells of |u|^2/mu                                     (low-wavenumber bound)
__global__ void CalculateMaxVelocity(float *d_u, float *d_v, float *d_mu, float *max_velocity, float *max_u2_mu,
                                     float *max_inv_dt, int nx, int ny) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < nx && j < ny) {
        // Calculate velocity magnitude at cell center
        float u_center = 0.5f * (d_u[j*(nx+1) + i] + d_u[j*(nx+1) + (i+1)]);
        float v_center = 0.5f * (d_v[j*nx + i] + d_v[(j+1)*nx + i]);
        float vel2 = u_center*u_center + v_center*v_center;
        float velocity_mag = sqrtf(vel2);
        float mu_c = d_mu[j*nx + i];
        float inv_dt = fabsf(u_center)/DX + fabsf(v_center)/DY + 2.0f*(mu_c/rho)*(1.0f/(DX*DX) + 1.0f/(DY*DY));

        // Update maximum velocity using atomic operation (valid for non-negative floats)
        atomicMax((int*)max_velocity, __float_as_int(velocity_mag));
        atomicMax((int*)max_u2_mu, __float_as_int(vel2 / mu_c));
        atomicMax((int*)max_inv_dt, __float_as_int(inv_dt));
    }
}

// Write VTK file for visualization
void writeVTKFile(const char* filename, float *u, float *v, float *p, float *visc, float *shear_rate, int nx, int ny, float dx, float dy) {
    FILE *fp = fopen(filename, "w");
    if (fp == NULL) {
        printf("Error opening file for VTK output\n");
        return;
    }

    // VTK header
    fprintf(fp, "# vtk DataFile Version 3.0\n");
    fprintf(fp, "Blood Flow Simulation\n");
    fprintf(fp, "ASCII\n");
    fprintf(fp, "DATASET RECTILINEAR_GRID\n");
    // [FIX: DIMENSIONS counts grid POINTS: (nx+1) x (ny+1) for nx x ny cells. The original wrote
    //  nx x ny points and (nx-1)(ny-1) cells, dropping the last cell column and row.]
    fprintf(fp, "DIMENSIONS %d %d 1\n", nx+1, ny+1);

    // Coordinate information
    fprintf(fp, "X_COORDINATES %d float\n", nx+1);
    for (int i = 0; i <= nx; i++) {
        fprintf(fp, "%f ", i*dx);
    }
    fprintf(fp, "\n");

    fprintf(fp, "Y_COORDINATES %d float\n", ny+1);
    for (int j = 0; j <= ny; j++) {
        fprintf(fp, "%f ", j*dy);
    }
    fprintf(fp, "\n");

    fprintf(fp, "Z_COORDINATES 1 float\n");
    fprintf(fp, "0.0\n");

    // Cell-centered data
    fprintf(fp, "CELL_DATA %d\n", nx*ny);

    // Pressure field
    fprintf(fp, "SCALARS pressure float 1\n");
    fprintf(fp, "LOOKUP_TABLE default\n");
    for (int j = 0; j < ny; j++) {
        for (int i = 0; i < nx; i++) {
            fprintf(fp, "%f\n", p[j*nx + i]);
        }
    }

    // Velocity magnitude
    fprintf(fp, "SCALARS velocity_magnitude float 1\n");
    fprintf(fp, "LOOKUP_TABLE default\n");
    for (int j = 0; j < ny; j++) {
        for (int i = 0; i < nx; i++) {
            // Interpolate u and v to cell centers
            float u_center = 0.5f * (u[j*(nx+1) + i] + u[j*(nx+1) + (i+1)]);
            float v_center = 0.5f * (v[j*nx + i] + v[(j+1)*nx + i]);
            float vel_mag = sqrtf(u_center*u_center + v_center*v_center);
            fprintf(fp, "%f\n", vel_mag);
        }
    }

    // [ADDED] viscosity and shear rate (the host copies were made but never written)
    fprintf(fp, "SCALARS viscosity float 1\n");
    fprintf(fp, "LOOKUP_TABLE default\n");
    for (int j = 0; j < ny; j++) for (int i = 0; i < nx; i++) fprintf(fp, "%e\n", visc[j*nx + i]);
    fprintf(fp, "SCALARS shear_rate float 1\n");
    fprintf(fp, "LOOKUP_TABLE default\n");
    for (int j = 0; j < ny; j++) for (int i = 0; i < nx; i++) fprintf(fp, "%e\n", shear_rate[j*nx + i]);

    // Velocity vectors
    fprintf(fp, "VECTORS velocity float\n");
    for (int j = 0; j < ny; j++) {
        for (int i = 0; i < nx; i++) {
            // Interpolate u and v to cell centers
            float u_center = 0.5f * (u[j*(nx+1) + i] + u[j*(nx+1) + (i+1)]);
            float v_center = 0.5f * (v[j*nx + i] + v[(j+1)*nx + i]);
            fprintf(fp, "%f %f 0.0\n", u_center, v_center);
        }
    }

    fclose(fp);
    printf("Saved VTK file: %s\n", filename);
}

// [ADDED] cell-centred final fields (x y u v p mu gamma) and wall shear stress for post-processing
void writeFinalFields(float *u, float *v, float *p, float *visc, float *shear_rate, int nx, int ny) {
    FILE *fp = fopen("final_fields.dat", "w");
    fprintf(fp, "# x[m] y[m] u[m/s] v[m/s] p[Pa] mu[Pa.s] gamma_dot[1/s]  (cell centres, %d x %d)\n", nx, ny);
    for (int j = 0; j < ny; j++)
        for (int i = 0; i < nx; i++) {
            float uc = 0.5f * (u[j*(nx+1) + i] + u[j*(nx+1) + (i+1)]);
            float vc = 0.5f * (v[j*nx + i] + v[(j+1)*nx + i]);
            fprintf(fp, "%.7e %.7e %.7e %.7e %.7e %.7e %.7e\n", (i + 0.5f)*DX, (j + 0.5f)*DY, uc, vc,
                    p[j*nx + i], visc[j*nx + i], shear_rate[j*nx + i]);
        }
    fclose(fp);
    // wall shear stress: wall gradient implied by the no-slip ghost value u(-1) = -u(0), i.e. 2 u_0/DY with
    // u_0 the cell-centre velocity at DY/2 (the wall flux of the finite-volume scheme; exact for the discrete
    // Poiseuille solution); tau_w = mu(gamma_w) * gamma_w
    fp = fopen("wall_shear_stress.csv", "w");
    fprintf(fp, "x_m,tau_bottom_Pa,tau_top_Pa\n");
    for (int i = 0; i < nx; i++) {
        float ub0 = 0.5f * (u[i] + u[i+1]);
        float ut0 = 0.5f * (u[(ny-1)*(nx+1) + i] + u[(ny-1)*(nx+1) + i+1]);
        float gb = 2.0f * ub0 / DY, gt = 2.0f * ut0 / DY;
        fprintf(fp, "%.7e,%.7e,%.7e\n", (i + 0.5f)*DX, carreau(fabsf(gb))*gb, carreau(fabsf(gt))*gt);
    }
    fclose(fp);
}

int main(int argc, char **argv) {
    // Initialize variables and parameters
    float *h_u, *h_v, *h_p, *h_visc, *h_shear_rate, *h_div;
    float *d_u, *d_v, *d_p, *d_visc, *d_shear_rate;
    float *d_u_star, *d_v_star, *d_p_prime, *d_divergence;
    float *d_u_old, *d_v_old;
    float *d_residual, *d_max_velocity, *d_max_u2_mu, *d_max_inv_dt, *d_col_res, *d_col_corr, *d_work;   // [ADDED]

    // Timing variables
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Allocate host memory
    h_u = (float*)malloc(sizeof(float) * (NX+1) * NY);
    h_v = (float*)malloc(sizeof(float) * NX * (NY+1));
    h_p = (float*)malloc(sizeof(float) * NX * NY);
    h_visc = (float*)malloc(sizeof(float) * NX * NY);
    h_shear_rate = (float*)malloc(sizeof(float) * NX * NY);
    h_div = (float*)malloc(sizeof(float) * NX * NY);

    // Allocate device memory
    cudaMalloc((void**)&d_u, sizeof(float) * (NX+1) * NY);
    cudaMalloc((void**)&d_v, sizeof(float) * NX * (NY+1));
    cudaMalloc((void**)&d_p, sizeof(float) * NX * NY);
    cudaMalloc((void**)&d_visc, sizeof(float) * NX * NY);
    cudaMalloc((void**)&d_shear_rate, sizeof(float) * NX * NY);

    // Allocate memory for intermediate values
    cudaMalloc((void**)&d_u_star, sizeof(float) * (NX+1) * NY);
    cudaMalloc((void**)&d_v_star, sizeof(float) * NX * (NY+1));
    cudaMalloc((void**)&d_p_prime, sizeof(float) * NX * NY);
    cudaMalloc((void**)&d_divergence, sizeof(float) * NX * NY);

    // Allocate memory for old values (needed for temporal discretization)
    cudaMalloc((void**)&d_u_old, sizeof(float) * (NX+1) * NY);
    cudaMalloc((void**)&d_v_old, sizeof(float) * NX * (NY+1));

    // [FIX: reduction scalars and solver work arrays allocated once (the original called cudaMalloc/
    //  cudaFree inside the time loop and inside every pressure solve)]
    cudaMalloc((void**)&d_residual, sizeof(float));
    cudaMalloc((void**)&d_max_velocity, sizeof(float));
    cudaMalloc((void**)&d_max_u2_mu, sizeof(float));
    cudaMalloc((void**)&d_max_inv_dt, sizeof(float));
    cudaMalloc((void**)&d_col_res, sizeof(float) * NX);
    cudaMalloc((void**)&d_col_corr, sizeof(float) * NX);
    cudaMalloc((void**)&d_work, sizeof(float) * NX);

    // Set execution configuration
    // [FIX: the launch grid must cover i = 0..NX and j = 0..NY (outlet u-faces, top-wall v-faces);
    //  sizing it with NX, NY only works by accident when NX, NY are not multiples of 16]
    dim3 blockSize(16, 16);
    dim3 gridSize((NX + 1 + blockSize.x - 1) / blockSize.x,
                  (NY + 1 + blockSize.y - 1) / blockSize.y);

    // Initialize flow field on device
    Initialize<<<gridSize, blockSize>>>(d_u, d_v, d_p, d_visc, d_shear_rate, del_P, NX, NY);
    cudaDeviceSynchronize();

    // Simulation parameters
    double physicalTime = 5.0;        // Total physical time to simulate (seconds)
    double currentTime = 0.0;         // Current physical time [CHANGED: double, so that ~1e5 additions of
                                      //  dt ~ 4e-5 s do not accumulate float round-off]
    int timeStep = 0;                 // Current time step
    float cfl = 0.5f;                 // Target CFL number
    float maxResidual = 1.0f;         // Initial residual
    float convergenceTolerance = 1e-5f; // Convergence criteria
    int maxIterations = 1;            // [FIX: was 1000 "SIMPLE iterations". Each pass of that loop
                                      //  re-predicted from the CURRENT u (not u^n) with the full dt, i.e.
                                      //  advanced the flow by one more explicit step, while the clock was
                                      //  advanced only once: up to 1000 dt per reported dt. One predictor-
                                      //  projection per time step is the explicit fractional-step method.]
    int outputInterval = 1000;        // VTK output every N time steps
    float omega = 1.85f;              // [CHANGED: SOR factor was 1.5 (inside the loop). After the block
                                      //  correction the slowest error modes vary across the 35-cell width;
                                      //  omega_opt = 2/(1+sin(pi/35)) ~ 1.84]
    int maxPoissonIter = 1000;        // Maximum SOR sweeps per pressure solve (as in the original)
    int blockCorrection = 1;          // [ADDED] 1 = block correction on (0 reproduces point SOR only)
    int logInterval = 100;            // [ADDED] history (CSV) every N steps

    // [CHANGED: parameters from the command line instead of scanf, so that runs can be scripted]
    if (argc > 1) physicalTime = atof(argv[1]);
    if (argc > 2) outputInterval = atoi(argv[2]);
    if (argc > 3) omega = atof(argv[3]);
    if (argc > 4) maxPoissonIter = atoi(argv[4]);
    if (argc > 5) blockCorrection = atoi(argv[5]);
    printf("Grid %d x %d cells (%.4f m x %.4f m), t_end = %.4f s, VTK every %d steps, SOR omega = %.3f, "
           "max sweeps = %d, block correction = %d\n", NX, NY, NX*DX, NY*DY, physicalTime, outputInterval,
           omega, maxPoissonIter, blockCorrection);

    // Calculate adaptive time step based on CFL condition
    // [FIX: the original used only the CFL step 0.5*DX/U = 1.28e-4 s. Explicit Euler for the momentum
    //  equation is stable only if, cell by cell,
    //     dt * ( |u|/DX + |v|/DY + 2 nu (1/DX^2 + 1/DY^2) ) <= 1     (high wavenumbers: diffusion+convection)
    //     dt * |u|^2 / (2 nu) <= 1                                   (low wavenumbers, FTCS-type)
    //  With mu = mu0 = 0.056 Pa s the first bound is 3.7e-5 s; 1.28e-4 s is 3.5 times too large. The
    //  pure diffusion limit rho DX^2/(4 mu0) = 4.7e-5 s is not enough either: at that value the
    //  (upwind-biased) QUICK dissipation pushes the cell-to-cell oscillation mode to |G| = 1 + CFL > 1.]
    float maxVelocity = 1.5f * U;  // Initial estimate (peak of the inlet profile)
    float safety = 0.9f;           // [ADDED]
    float adaptiveDt = safety / (maxVelocity/DX + 2.0f*(mu0/rho)*(1.0f/(DX*DX) + 1.0f/(DY*DY)));
    adaptiveDt = fminf(adaptiveDt, cfl * DX / maxVelocity);
    printf("Initial time step: %e seconds\n", adaptiveDt);

    // Create output file
    FILE *outfile = fopen("blood_flow_results.csv", "w");
    fprintf(outfile, "Time,TimeStep,PoissonSweeps,RMS_dudt,MaxVelocity,MaxPressure,dt,MaxDivRel,Qout_over_Qin\n");

    // Start timing
    cudaEventRecord(start, 0);
    int poissonIters = 0;
    float h_max_velocity = maxVelocity, h_max_u2_mu = 0.0f, h_max_inv_dt = 0.0f;

    // Time advancement loop
    while (currentTime < physicalTime - 1e-9) {
        timeStep++;

        // Store previous time step values for temporal discretization
        cudaMemcpy(d_u_old, d_u, sizeof(float) * (NX+1) * NY, cudaMemcpyDeviceToDevice);
        cudaMemcpy(d_v_old, d_v, sizeof(float) * NX * (NY+1), cudaMemcpyDeviceToDevice);

        // Predictor-projection (one pass per time step, see maxIterations)
        int iterations = 0;
        maxResidual = 1.0f;

        while (iterations < maxIterations && maxResidual > convergenceTolerance) {
            // Step 1: Apply boundary conditions
            ApplyBoundaryConditions<<<gridSize, blockSize>>>(d_u, d_v, d_p, 0.0f, NX, NY);

            // Step 2: Momentum prediction
            Xmomentum_solver<<<gridSize, blockSize>>>(d_u, d_v, d_p, d_visc, d_u_star, adaptiveDt, DX, DY, rho, NX, NY);
            YMomentum_solver<<<gridSize, blockSize>>>(d_u, d_v, d_p, d_v_star, adaptiveDt, DX, DY, rho, d_visc, NX, NY);

            // Step 3: Calculate divergence
            Divergence<<<gridSize, blockSize>>>(d_u_star, d_v_star, d_divergence, NX, NY);

            // Step 4: Solve pressure correction equation
            SolvePressurePoisson(d_divergence, d_p_prime, NX, NY, omega, convergenceTolerance/10.0f, adaptiveDt,
                                 maxPoissonIter, blockCorrection, d_residual, d_col_res, d_col_corr, d_work, &poissonIters);

            // Step 5: Correct velocities and pressure
            CorrectVelocities<<<gridSize, blockSize>>>(d_u, d_v, d_u_star, d_v_star, d_p, d_p_prime, adaptiveDt, NX, NY);
            ApplyBoundaryConditions<<<gridSize, blockSize>>>(d_u, d_v, d_p, 0.0f, NX, NY);   // [ADDED] outlet face

            // Step 6: Update viscosity using Bird-Carreau model
            Shear_rate<<<gridSize, blockSize>>>(d_u, d_v, d_visc, d_shear_rate, NX, NY);

            // Velocity change over the step  [CHANGED: reported as RMS du/dt]
            cudaMemset(d_residual, 0, sizeof(float));
            CalculateResidual<<<gridSize, blockSize>>>(d_u, d_u_old, d_v, d_v_old, d_residual, NX, NY);
            float h_res;
            cudaMemcpy(&h_res, d_residual, sizeof(float), cudaMemcpyDeviceToHost);
            maxResidual = sqrtf(h_res / ((NX+1)*NY + NX*(NY+1))) / adaptiveDt;
            iterations++;
        }

        // Update current time
        currentTime += adaptiveDt;
        float dtTaken = adaptiveDt;   // [ADDED] for the history file

        // Calculate maximum velocity and the stability indicators for adaptive time stepping
        cudaMemset(d_max_velocity, 0, sizeof(float));
        cudaMemset(d_max_u2_mu, 0, sizeof(float));
        cudaMemset(d_max_inv_dt, 0, sizeof(float));
        CalculateMaxVelocity<<<gridSize, blockSize>>>(d_u, d_v, d_visc, d_max_velocity, d_max_u2_mu, d_max_inv_dt, NX, NY);
        cudaMemcpy(&h_max_velocity, d_max_velocity, sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(&h_max_u2_mu, d_max_u2_mu, sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(&h_max_inv_dt, d_max_inv_dt, sizeof(float), cudaMemcpyDeviceToHost);

        // Update time step based on CFL condition
        // [FIX: dt = min(CFL, high-wavenumber, low-wavenumber) limits, evaluated cell by cell (see above)]
        float dtNew = safety / h_max_inv_dt;
        if (h_max_velocity > 0.01f) {  // Avoid division by very small values
            dtNew = fminf(dtNew, cfl * DX / h_max_velocity);
        }
        if (h_max_u2_mu > 0.0f) dtNew = fminf(dtNew, safety * 2.0f / (rho * h_max_u2_mu));

        // Limit time step growth for stability
        adaptiveDt = fminf(dtNew, 1.5f * adaptiveDt);   // [FIX: was fminf(adaptiveDt, 1.5f*adaptiveDt), a no-op]

        // Adjust final time step to exactly reach physicalTime
        // [FIX: the remainder is split over the last two steps. A last step of arbitrary size (possibly
        //  ~1e-9 s) turns the small residual divergence into a pressure correction ~ rho*div*DX^2/dt that
        //  corrupts the final pressure field.]
        double remaining = physicalTime - currentTime;
        if (remaining < 2.0 * adaptiveDt) {
            adaptiveDt = (float)((remaining > adaptiveDt) ? 0.5 * remaining : remaining);
        }

        bool lastStep = (currentTime >= physicalTime - 1e-9);
        // [ADDED] history every logInterval steps: pressure maximum, mass conservation
        if (timeStep % logInterval == 0 || timeStep % outputInterval == 0 || lastStep) {
            cudaMemcpy(h_u, d_u, sizeof(float) * (NX+1) * NY, cudaMemcpyDeviceToHost);
            cudaMemcpy(h_p, d_p, sizeof(float) * NX * NY, cudaMemcpyDeviceToHost);
            Divergence<<<gridSize, blockSize>>>(d_u, d_v, d_divergence, NX, NY);
            cudaMemcpy(h_div, d_divergence, sizeof(float) * NX * NY, cudaMemcpyDeviceToHost);

            // Calculate maximum pressure
            float maxPressure = 0.0f, maxDiv = 0.0f, qin = 0.0f, qout = 0.0f;
            for (int k = 0; k < NX*NY; k++) {
                if (fabs(h_p[k]) > maxPressure) maxPressure = fabs(h_p[k]);
                if ((k % NX) < NX-1 && fabs(h_div[k]) > maxDiv) maxDiv = fabs(h_div[k]);
            }
            for (int j = 0; j < NY; j++) { qin += h_u[j*(NX+1)] * DY; qout += h_u[j*(NX+1) + NX] * DY; }

            // Write to output file
            fprintf(outfile, "%.6f,%d,%d,%.6e,%.6f,%.6f,%.6e,%.6e,%.6f\n",
                    currentTime, timeStep, poissonIters, maxResidual, h_max_velocity, maxPressure, dtTaken,
                    maxDiv * DX / U, qout / qin);

            // Output results at specified intervals
            if (timeStep % outputInterval == 0 || lastStep) {
                printf("Time: %.4f s, Step: %d, Poisson sweeps: %d, RMS du/dt: %.6e, Max Velocity: %.4f m/s, "
                       "max|div|DX/U: %.2e, Qout/Qin: %.6f\n", currentTime, timeStep, poissonIters, maxResidual,
                       h_max_velocity, maxDiv * DX / U, qout / qin);

                // Copy data back to host for output
                cudaMemcpy(h_v, d_v, sizeof(float) * NX * (NY+1), cudaMemcpyDeviceToHost);
                cudaMemcpy(h_visc, d_visc, sizeof(float) * NX * NY, cudaMemcpyDeviceToHost);
                cudaMemcpy(h_shear_rate, d_shear_rate, sizeof(float) * NX * NY, cudaMemcpyDeviceToHost);

                // Write field data to VTK file for visualization
                char filename[256];
                sprintf(filename, "blood_flow_t%.4f.vtk", currentTime);
                writeVTKFile(filename, h_u, h_v, h_p, h_visc, h_shear_rate, NX, NY, DX, DY);
            }
        }
    }

    // Stop timing and report
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    cudaError_t err = cudaGetLastError();   // [ADDED]
    if (err != cudaSuccess) printf("CUDA error: %s\n", cudaGetErrorString(err));
    printf("Total simulation time: %.2f seconds (%d time steps, %.3f ms per step)\n",
           elapsedTime/1000.0f, timeStep, elapsedTime/timeStep);

    // [ADDED] final cell-centred fields and wall shear stress
    writeFinalFields(h_u, h_v, h_p, h_visc, h_shear_rate, NX, NY);

    // Close output file
    fclose(outfile);

    // Clean up
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Free host memory
    free(h_u);
    free(h_v);
    free(h_p);
    free(h_visc);
    free(h_shear_rate);
    free(h_div);

    // Free device memory
    cudaFree(d_u);
    cudaFree(d_v);
    cudaFree(d_p);
    cudaFree(d_visc);
    cudaFree(d_shear_rate);
    cudaFree(d_u_star);
    cudaFree(d_v_star);
    cudaFree(d_p_prime);
    cudaFree(d_divergence);
    cudaFree(d_u_old);
    cudaFree(d_v_old);
    cudaFree(d_residual);
    cudaFree(d_max_velocity);
    cudaFree(d_max_u2_mu);
    cudaFree(d_max_inv_dt);
    cudaFree(d_col_res);
    cudaFree(d_col_corr);
    cudaFree(d_work);

    return 0;
}
