// Blood flow through arteries (LAD) artery analysis
// The flow is assumed to be unsteady, incompressible, laminar, non-Newtonian, and fully developed

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <float.h>

// Defining the variables for the 2D flow simulation
#define rho 1.06e3f   // density of blood in kg/m^3
#define L  0.14f      // length of the artery in m
#define W  0.0035f    // width of the artery in m
#define U  0.39f      // velocity of the blood in m/s in x direction in the artery
#define V  0.0f       // velocity of the blood in m/s in y direction in the artery
#define DX 0.0001f    // length of the cell in m for x direction (100micrometers)
#define DY 0.0001f    // length of the cell in m for y direction (100micrometers)
#define dt 1.0e-6f    // time step in seconds - REDUCED for stability
#define NX ((int)(L/DX)+1)   // number of cells in x direction
#define NY ((int)(W/DY)+1)   // number of cells in y direction
#define N  (NX*NY)           // total number of cells
#define del_P 2.8863e3f      // initial pressure difference in Pa using the Poiseuille equation

// Bird-Carreau model parameters
#define mu0    0.056f   // dynamic viscosity of blood in Pa.s at zero 
#define mu_inf 0.0035f  // dynamic viscosity of blood in Pa.s at infinite
#define lambda 3.13f    // time constant in seconds
#define n      0.6f     // flow index for shear thinning fluid 
#define a      2.0f     // bird-carreau model parameter 

// Function prototypes
void Initialize(float *u, float *v, float *p, float *visc, float *shear_rate, float p_init, int nx, int ny);
void ApplyBoundaryConditions(float *u, float *v, float *p, float p_out, int nx, int ny);
void Shear_rate(float *u, float *v, float *visc, float *shear_rate, int nx, int ny);
void Xmomentum_solver(float *u_x, float *u_y, float *p, float *visc, float *u_x_new, float dt_val, float dx, float dy, float rho_val, int nx, int ny);
void YMomentum_solver(float *u_x, float *u_y, float *p, float *v_new, float dt_val, float dx, float dy, float rho_val, float *visc, int nx, int ny);
void Divergence(float *u_x_new, float *v_new, float *divergence, int nx, int ny);
int SolvePressurePoisson(float *divergence, float *p, float *p_new, int nx, int ny, float omega, float tolerance);
void CorrectVelocities(float *u, float *v, float *u_star, float *v_star, float *p, int nx, int ny);
float CalculateResidual(float *d_u, float *d_u_old, float *d_v, float *d_v_old, int nx, int ny);
float CalculateMaxVelocity(float *d_u, float *d_v, int nx, int ny);
void writeVTKFile(const char* filename, float *u, float *v, float *p, int nx, int ny, float dx, float dy);
void *safe_malloc(size_t size);
float clamp(float value, float min, float max);

// Safe memory allocation with error checking
void *safe_malloc(size_t size) {
    void *ptr = malloc(size);
    if (ptr == NULL) {
        fprintf(stderr, "Error: Memory allocation failed for size %zu bytes\n", size);
        exit(EXIT_FAILURE);
    }
    return ptr;
}

// Clamp function to prevent extreme values
float clamp(float value, float min, float max) {
    if (isnan(value)) return 0.0f;
    if (value < min) return min;
    if (value > max) return max;
    return value;
}

// Initialize flow field with initial conditions
void Initialize(float *u, float *v, float *p, float *visc, float *shear_rate, float p_init, int nx, int ny) {
    printf("Initializing flow field with %d x %d cells...\n", nx, ny);
    
    // Initialize pressure and viscosity at cell centers
    for (int j = 0; j < ny; j++) {
        for (int i = 0; i < nx; i++) {
            int idx = i + j*nx;  // Cell-centered index
            
            // Linear pressure gradient from inlet to outlet
            p[idx] = p_init * (1.0f - (float)i/(nx-1));
            
            // Initialize with zero-shear viscosity
            visc[idx] = mu0;
            
            // Initialize shear rate to zero
            shear_rate[idx] = 0.0f;
        }
    }
    
    // Initialize x-velocity at east/west faces (staggered grid)
    for (int j = 0; j < ny; j++) {
        for (int i = 0; i <= nx; i++) {
            int idx_u = i + j*(nx+1);  // Face-centered index for u
            
            // Parabolic velocity profile at inlet (i=0)
            if (i == 0) {
                // y position normalized to [-1, 1]
                float y_norm = 2.0f * (float)j/(ny-1) - 1.0f;
                // Parabolic profile: u = U_max * (1 - y^2)
                u[idx_u] = U * (1.0f - y_norm*y_norm);
            } 
            // Zero gradient at outlet (i=nx)
            else if (i == nx) {
                u[idx_u] = u[idx_u-1];
            }
            // Interior points - initialize with interpolated values
            else {
                float x_pos = (float)i/nx;
                // Smooth transition from inlet profile to developed flow
                float y_norm = 2.0f * (float)j/(ny-1) - 1.0f;
                u[idx_u] = U * (1.0f - y_norm*y_norm) * (1.0f - 0.5f*x_pos);
            }
        }
    }
    
    // Initialize y-velocity at north/south faces (staggered grid)
    for (int j = 0; j <= ny; j++) {
        for (int i = 0; i < nx; i++) {
            int idx_v = i + j*nx;  // Face-centered index for v
            v[idx_v] = 0.0f;  // Initialize to zero (no y-component initially)
        }
    }
    
    printf("Flow field initialized successfully.\n");
}

// Apply boundary conditions for staggered grid
void ApplyBoundaryConditions(float *u, float *v, float *p, float p_out, int nx, int ny) {
    // Inlet boundary (west face, x = 0)
    for (int j = 0; j < ny; j++) {
        // Parabolic velocity profile at inlet
        float y_norm = 2.0f * (float)j/(ny-1) - 1.0f;
        u[j*(nx+1)] = U * (1.0f - y_norm*y_norm);
        
        // No y-component at inlet
        v[j*nx] = 0.0f;
        
        // Zero gradient for pressure at inlet
        p[j*nx] = p[j*nx + 1];
    }
    
    // Outlet boundary (east face, x = nx)
    for (int j = 0; j < ny; j++) {
        // Zero gradient for velocity at outlet
        u[j*(nx+1) + nx] = u[j*(nx+1) + (nx-1)];
        
        // Fixed pressure at outlet
        p[j*nx + (nx-1)] = p_out;
    }
    
    // Wall boundaries (south face, y = 0)
    for (int i = 0; i < nx; i++) {
        // No-slip condition at bottom wall
        v[i] = 0.0f;
    }
    
    // Wall boundaries (north face, y = ny)
    for (int i = 0; i < nx; i++) {
        // No-slip condition at top wall
        v[i + ny*nx] = 0.0f;
    }
    
    // No-slip condition for u at walls
    for (int i = 0; i <= nx; i++) {
        // Bottom wall
        u[0*(nx+1) + i] = 0.0f;
        // Top wall
        u[(ny-1)*(nx+1) + i] = 0.0f;
    }
}

// Calculate shear rate and update viscosity using Bird-Carreau model
void Shear_rate(float *u, float *v, float *visc, float *shear_rate, int nx, int ny) {
    for (int j = 1; j < ny-1; j++) {
        for (int i = 1; i < nx-1; i++) {
            // For staggered grid, u is at (i+1/2,j) and v is at (i,j+1/2)
            // Calculate velocity gradients at cell center (i,j)
            float du_dx = (u[j*(nx+1) + (i+1)] - u[j*(nx+1) + i]) / DX;
            float dv_dy = (v[(j+1)*nx + i] - v[j*nx + i]) / DY;
            
            // Cross derivatives require averaging
            float du_dy = (u[(j+1)*(nx+1) + i] + u[(j+1)*(nx+1) + (i+1)] -
                          u[j*(nx+1) + i] - u[j*(nx+1) + (i+1)]) / (2.0f * DY);
            float dv_dx = (v[j*nx + (i+1)] + v[(j+1)*nx + (i+1)] - 
                           v[j*nx + i] - v[(j+1)*nx + i]) / (2.0f * DX);
            
            // Calculate shear rate
            float sr = sqrtf(2.0f * (du_dx*du_dx + dv_dy*dv_dy) + (du_dy + dv_dx)*(du_dy + dv_dx));
            
            // Add small constant to avoid division by zero
            sr = fmaxf(sr, 1e-6f);
            
            // Store at cell center
            int idx = j*nx + i;
            shear_rate[idx] = sr;
            
            // Update viscosity using Bird-Carreau model
            visc[idx] = mu_inf + (mu0 - mu_inf) * powf(1.0f + powf(lambda * sr, a), (n-1.0f)/a);
            
            // Clamp viscosity to reasonable values
            visc[idx] = clamp(visc[idx], mu_inf, mu0);
        }
    }
    
    // Set boundary values to zero or a safe default
    for (int j = 0; j < ny; j++) {
        for (int i = 0; i < nx; i++) {
            if (i == 0 || i == nx-1 || j == 0 || j == ny-1) {
                int idx = j*nx + i;
                shear_rate[idx] = 0.0f;
                visc[idx] = mu0;
            }
        }
    }
}

// X-momentum equation solver 
void Xmomentum_solver(float *u_x, float *u_y, float *p, float *visc, float *u_x_new, float dt_val, float dx, float dy, float rho_val, int nx, int ny) {
    // First copy u_x to u_x_new to handle boundary cells
    memcpy(u_x_new, u_x, sizeof(float) * (nx+1) * ny);
    
    for (int j = 1; j < ny-1; j++) {
        for (int i = 1; i < nx; i++) {
            // For staggered grid, u_x is stored at (i+1/2, j)
            int idx_u = j*(nx+1) + i;
            
            // 1. Calculate convection term using central difference for stability
            float u_e = 0.5f * (u_x[j*(nx+1) + i] + u_x[j*(nx+1) + (i+1)]);
            float u_w = 0.5f * (u_x[j*(nx+1) + (i-1)] + u_x[j*(nx+1) + i]);
            float u_n = 0.5f * (u_x[(j+1)*(nx+1) + i] + u_x[j*(nx+1) + i]);
            float u_s = 0.5f * (u_x[(j-1)*(nx+1) + i] + u_x[j*(nx+1) + i]);
            
            // Interpolate v-velocities to u-cell faces
            float v_n = 0.25f * (u_y[(j+1)*nx + i] + u_y[(j+1)*nx + (i-1)] + 
                               u_y[j*nx + i] + u_y[j*nx + (i-1)]);
            float v_s = 0.25f * (u_y[j*nx + i] + u_y[j*nx + (i-1)] + 
                               u_y[(j-1)*nx + i] + u_y[(j-1)*nx + (i-1)]);
            
            // Calculate mass fluxes through cell faces
            float Fe = u_e * dy;
            float Fw = u_w * dy;
            float Fn = v_n * dx;
            float Fs = v_s * dx;
            
            // Central difference for convection
            float convection_term = -(Fe * u_x[j*(nx+1) + (i+1)] - Fw * u_x[j*(nx+1) + (i-1)] + 
                                     Fn * u_x[(j+1)*(nx+1) + i] - Fs * u_x[(j-1)*(nx+1) + i]) / (2.0f * dx * dy);
            
            // 2. Calculate diffusion term using central difference
            // Interpolate viscosity to u-cell faces
            float mu_e = 0.5f * (visc[j*nx + i] + visc[j*nx + (i+1)]);
            float mu_w = 0.5f * (visc[j*nx + (i-1)] + visc[j*nx + i]);
            float mu_n = 0.5f * (visc[j*nx + i] + visc[(j+1)*nx + i]);
            float mu_s = 0.5f * (visc[(j-1)*nx + i] + visc[j*nx + i]);

            // Calculate diffusion coefficients
            float De = mu_e * dy/dx;
            float Dw = mu_w * dy/dx;
            float Dn = mu_n * dx/dy;
            float Ds = mu_s * dx/dy;
            
            // Calculate diffusion term using coefficients
            float diffusion = De * (u_x[j*(nx+1) + (i+1)] - u_x[j*(nx+1) + i]) - 
                              Dw * (u_x[j*(nx+1) + i] - u_x[j*(nx+1) + (i-1)]) + 
                              Dn * (u_x[(j+1)*(nx+1) + i] - u_x[j*(nx+1) + i]) - 
                              Ds * (u_x[j*(nx+1) + i] - u_x[(j-1)*(nx+1) + i]);
            
            // 3. Calculate pressure gradient (staggered grid - pressure at cell centers)
            float pressure_gradient_term = (p[j*nx + i] - p[j*nx + (i-1)]) / dx;
            
            // 4. Update u_x using the momentum equation
            u_x_new[idx_u] = u_x[idx_u] + dt_val * (convection_term + diffusion/(dx*dy*rho_val) - pressure_gradient_term/rho_val);
            
            // Clamp velocity to prevent extreme values
            u_x_new[idx_u] = clamp(u_x_new[idx_u], -10.0f*U, 10.0f*U);
        }
    }
}

// Y-momentum equation solver
void YMomentum_solver(float *u_x, float *u_y, float *p, float *v_new, float dt_val, float dx, float dy, float rho_val, float *visc, int nx, int ny) {
    // First copy u_y to v_new to handle boundary cells
    memcpy(v_new, u_y, sizeof(float) * nx * (ny+1));
    
    for (int j = 1; j < ny; j++) {
        for (int i = 1; i < nx-1; i++) {
            // For staggered grid, v is at (i,j+1/2)
            int idx = i + j*nx;
            
            // Interpolate u-velocities to v-cell faces (east and west)
            float u_e = 0.25f * (u_x[i + j*(nx+1)] + u_x[(i+1) + j*(nx+1)] + 
                               u_x[i + (j-1)*(nx+1)] + u_x[(i+1) + (j-1)*(nx+1)]);
            float u_w = 0.25f * (u_x[(i-1) + j*(nx+1)] + u_x[i + j*(nx+1)] + 
                               u_x[(i-1) + (j-1)*(nx+1)] + u_x[i + (j-1)*(nx+1)]);
            
            // Interpolate v-velocities to v-cell faces (north and south)
            float v_n = 0.5f * (u_y[i + j*nx] + u_y[i + (j+1)*nx]);
            float v_s = 0.5f * (u_y[i + (j-1)*nx] + u_y[i + j*nx]);
            
            // Calculate mass fluxes
            float Fe = u_e * dy;
            float Fw = u_w * dy;
            float Fn = v_n * dx;
            float Fs = v_s * dx;
            
            // Central difference for convection
            float convection_term = -(Fe * u_y[(i+1) + j*nx] - Fw * u_y[(i-1) + j*nx] + 
                                     Fn * u_y[i + (j+1)*nx] - Fs * u_y[i + (j-1)*nx]) / (2.0f * dx * dy);
            
            // Viscosity interpolation - corrected indexing for staggered grid
            float mu_e = 0.5f * (visc[i + j*nx] + visc[(i+1) + j*nx]);
            float mu_w = 0.5f * (visc[(i-1) + j*nx] + visc[i + j*nx]);
            float mu_n = 0.5f * (visc[i + j*nx] + visc[i + (j+1)*nx]);
            float mu_s = 0.5f * (visc[i + (j-1)*nx] + visc[i + j*nx]);
            
            // Calculate diffusion coefficients
            float De = mu_e * dy/dx;
            float Dw = mu_w * dy/dx;
            float Dn = mu_n * dx/dy;
            float Ds = mu_s * dx/dy;
            
            // Calculate diffusion term using coefficients
            float diffusion = De * (u_y[(i+1) + j*nx] - u_y[i + j*nx]) - 
                              Dw * (u_y[i + j*nx] - u_y[(i-1) + j*nx]) + 
                              Dn * (u_y[i + (j+1)*nx] - u_y[i + j*nx]) - 
                              Ds * (u_y[i + j*nx] - u_y[i + (j-1)*nx]);
            
            // Pressure gradient - corrected for v-velocity at (i,j+1/2)
            float pressure_gradient_term = (p[i + j*nx] - p[i + (j-1)*nx]) / dy;
            
            // Update v-velocity
            v_new[idx] = u_y[idx] + dt_val * (convection_term + diffusion/(dx*dy*rho_val) - pressure_gradient_term/rho_val);
            
            // Clamp velocity to prevent extreme values
            v_new[idx] = clamp(v_new[idx], -10.0f*U, 10.0f*U);
        }
    }
}

// Calculate divergence for pressure correction
void Divergence(float *u_x_new, float *v_new, float *divergence, int nx, int ny) {
    // Initialize divergence to zero
    memset(divergence, 0, nx * ny * sizeof(float));
    
    for (int j = 1; j < ny-1; j++) {
        for (int i = 1; i < nx-1; i++) {
            // For staggered grid:
            // u_x is stored at (i+1/2,j) - east/west cell faces
            // v is stored at (i,j+1/2) - north/south cell faces
            
            // Calculate divergence using central difference
            float du_dx = (u_x_new[j*(nx+1) + (i+1)] - u_x_new[j*(nx+1) + i]) / DX;
            float dv_dy = (v_new[(j+1)*nx + i] - v_new[j*nx + i]) / DY;
            
            // Store divergence at cell center (i,j)
            divergence[j*nx + i] = du_dx + dv_dy;
        }
    }
}

// Pressure Poisson equation solver using Gauss-Seidel with SOR
void PoissonSolverGS(float *divergence, float *p, float *p_new, int nx, int ny, float omega, float *residual) {
    *residual = 0.0f;
    
    for (int j = 1; j < ny-1; j++) {
        for (int i = 1; i < nx-1; i++) {
            // Proper indexing for staggered grid (cell centers)
            int idx = j*nx + i;
            
            float source = -rho * divergence[idx] / dt;
            float p_old = p[idx];
            
            // Gauss-Seidel iteration for pressure correction
            float p_gs = 0.25f * (
                p_new[j*nx + (i+1)] + p_new[j*nx + (i-1)] + 
                p_new[(j+1)*nx + i] + p_new[(j-1)*nx + i] - 
                DX * DY * source
            );
            
            // Apply SOR (Successive Over-Relaxation)
            p_new[idx] = (1.0f - omega) * p_old + omega * p_gs;
            
            // Clamp pressure to prevent extreme values
            p_new[idx] = clamp(p_new[idx], 0.0f, del_P * 2.0f);
            
            // Calculate local residual for convergence check
            *residual += fabsf(p_new[idx] - p_old);
        }
    }
}

// Host-side wrapper for Gauss-Seidel iteration with convergence check
int SolvePressurePoisson(float *divergence, float *p, float *p_new, int nx, int ny, float omega, float tolerance) {
    float residual;
    int max_iter = 1000;  // Maximum iterations

    for (int iter = 0; iter < max_iter; ++iter) {
        // Reset residual
        residual = 0.0f;

        // Copy current pressure to p_new as initial guess
        memcpy(p_new, p, nx * ny * sizeof(float));

        // Perform one Gauss-Seidel iteration
        PoissonSolverGS(divergence, p, p_new, nx, ny, omega, &residual);

        // Check convergence
        residual /= (nx * ny);  // Normalize by grid size

        if (residual < tolerance) {
            memcpy(p, p_new, nx * ny * sizeof(float));
            return 1;  // Converged
        }

        // Update pressure for next iteration
        memcpy(p, p_new, nx * ny * sizeof(float));
    }

    return 0;  // Did not converge within max_iter
}

// Correct velocities using pressure correction
void CorrectVelocities(float *u, float *v, float *u_star, float *v_star, float *p, int nx, int ny) {
    // Correct u-velocity (staggered at i+1/2, j)
    for (int j = 1; j < ny-1; j++) {
        for (int i = 1; i < nx; i++) {
            // d-coefficient from momentum equation (simplified)
            float d_u = dt / rho;
            
            // Proper indexing for staggered grid
            int idx_u = j*(nx+1) + i;  // u at (i+1/2,j)
            
            // Pressure gradient between adjacent cell centers
            float dp_dx = (p[j*nx + i] - p[j*nx + (i-1)]) / DX;
            
            // Correct u-velocity using pressure gradient
            u[idx_u] = u_star[idx_u] - d_u * dp_dx;
            
            // Clamp velocity to prevent extreme values
            u[idx_u] = clamp(u[idx_u], -10.0f*U, 10.0f*U);
        }
    }
    
    // Correct v-velocity (staggered at i, j+1/2)
    for (int j = 1; j < ny; j++) {
        for (int i = 1; i < nx-1; i++) {
            // d-coefficient from momentum equation (simplified)
            float d_v = dt / rho;
            
            // Proper indexing for staggered grid
            int idx_v = j*nx + i;  // v at (i,j+1/2)
            
            // Pressure gradient between adjacent cell centers
            float dp_dy = (p[j*nx + i] - p[(j-1)*nx + i]) / DY;
            
            // Correct v-velocity using pressure gradient
            v[idx_v] = v_star[idx_v] - d_v * dp_dy;
            
            // Clamp velocity to prevent extreme values
            v[idx_v] = clamp(v[idx_v], -10.0f*U, 10.0f*U);
        }
    }
}

// Calculate residual between iterations
float CalculateResidual(float *d_u, float *d_u_old, float *d_v, float *d_v_old, int nx, int ny) {
    float residual = 0.0f;
    
    for (int j = 0; j < ny; j++) {
        for (int i = 0; i < nx+1; i++) {
            // Calculate u-velocity residual
            float u_diff = d_u[j*(nx+1) + i] - d_u_old[j*(nx+1) + i];
            residual += u_diff * u_diff;
        }
    }
    
    for (int j = 0; j < ny+1; j++) {
        for (int i = 0; i < nx; i++) {
            // Calculate v-velocity residual
            float v_diff = d_v[j*nx + i] - d_v_old[j*nx + i];
            residual += v_diff * v_diff;
        }
    }
    
    return sqrtf(residual);  // Return L2 norm of residual
}

// Calculate maximum velocity for CFL condition
float CalculateMaxVelocity(float *d_u, float *d_v, int nx, int ny) {
    float max_velocity = 0.0f;
    
    for (int j = 0; j < ny; j++) {
        for (int i = 0; i < nx; i++) {
            // Calculate velocity magnitude at cell center
            float u_center = 0.5f * (d_u[j*(nx+1) + i] + d_u[j*(nx+1) + (i+1)]);
            float v_center = 0.5f * (d_v[j*nx + i] + d_v[(j+1)*nx + i]);
            float velocity_mag = sqrtf(u_center*u_center + v_center*v_center);
            
            // Update maximum velocity
            if (velocity_mag > max_velocity) {
                max_velocity = velocity_mag;
            }
        }
    }
    
    return max_velocity;
}

// Write VTK file for visualization in ParaView
void writeVTKFile(const char* filename, float *u, float *v, float *p, int nx, int ny, float dx, float dy) {
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
    fprintf(fp, "DIMENSIONS %d %d 1\n", nx, ny);
    
    // Coordinate information
    fprintf(fp, "X_COORDINATES %d float\n", nx);
    for (int i = 0; i < nx; i++) {
        fprintf(fp, "%f ", i*dx);
    }
    fprintf(fp, "\n");
    
    fprintf(fp, "Y_COORDINATES %d float\n", ny);
    for (int j = 0; j < ny; j++) {
        fprintf(fp, "%f ", j*dy);
    }
    fprintf(fp, "\n");
    
    fprintf(fp, "Z_COORDINATES 1 float\n");
    fprintf(fp, "0.0\n");
    
    // Cell-centered data
    fprintf(fp, "CELL_DATA %d\n", (nx-1)*(ny-1));
    
    // Pressure field
    fprintf(fp, "SCALARS pressure float 1\n");
    fprintf(fp, "LOOKUP_TABLE default\n");
    for (int j = 0; j < ny-1; j++) {
        for (int i = 0; i < nx-1; i++) {
            fprintf(fp, "%f\n", p[j*nx + i]);
        }
    }
    
    // Velocity magnitude
    fprintf(fp, "SCALARS velocity_magnitude float 1\n");
    fprintf(fp, "LOOKUP_TABLE default\n");
    for (int j = 0; j < ny-1; j++) {
        for (int i = 0; i < nx-1; i++) {
            // Interpolate u and v to cell centers
            float u_center = 0.5f * (u[j*(nx+1) + i] + u[j*(nx+1) + (i+1)]);
            float v_center = 0.5f * (v[j*nx + i] + v[(j+1)*nx + i]);
            float vel_mag = sqrtf(u_center*u_center + v_center*v_center);
            fprintf(fp, "%f\n", vel_mag);
        }
    }
    
    // Velocity vectors
    fprintf(fp, "VECTORS velocity float\n");
    for (int j = 0; j < ny-1; j++) {
        for (int i = 0; i < nx-1; i++) {
            // Interpolate u and v to cell centers
            float u_center = 0.5f * (u[j*(nx+1) + i] + u[j*(nx+1) + (i+1)]);
            float v_center = 0.5f * (v[j*nx + i] + v[(j+1)*nx + i]);
            fprintf(fp, "%f %f 0.0\n", u_center, v_center);
        }
    }
    
    fclose(fp);
    printf("Saved VTK file: %s\n", filename);
}

int main() {
    // Print simulation parameters
    printf("Simulation parameters:\n");
    printf("  Domain size: %.4f x %.4f m\n", L, W);
    printf("  Grid size: %d x %d cells\n", NX, NY);
    printf("  Cell size: %.6f x %.6f m\n", DX, DY);
    printf("  Blood density: %.2f kg/m^3\n", rho);
    printf("  Inlet velocity: %.2f m/s\n", U);
    printf("  Time step: %.8f s\n", dt);
    
    // Initialize variables and parameters
    float *u, *v, *p, *visc, *shear_rate;
    float *u_star, *v_star, *p_prime, *divergence;
    float *u_old, *v_old;
    
    // Start timing
    clock_t start_time = clock();
    
    // Allocate memory with error checking
    u = (float*)safe_malloc(sizeof(float) * (NX+1) * NY);
    v = (float*)safe_malloc(sizeof(float) * NX * (NY+1));
    p = (float*)safe_malloc(sizeof(float) * NX * NY);
    visc = (float*)safe_malloc(sizeof(float) * NX * NY);
    shear_rate = (float*)safe_malloc(sizeof(float) * NX * NY);
    
    // Allocate memory for intermediate values
    u_star = (float*)safe_malloc(sizeof(float) * (NX+1) * NY);
    v_star = (float*)safe_malloc(sizeof(float) * NX * (NY+1));
    p_prime = (float*)safe_malloc(sizeof(float) * NX * NY);
    divergence = (float*)safe_malloc(sizeof(float) * NX * NY);
    
    // Allocate memory for old values (needed for temporal discretization)
    u_old = (float*)safe_malloc(sizeof(float) * (NX+1) * NY);
    v_old = (float*)safe_malloc(sizeof(float) * NX * (NY+1));
    
    // Initialize all arrays to zero first
    memset(u, 0, sizeof(float) * (NX+1) * NY);
    memset(v, 0, sizeof(float) * NX * (NY+1));
    memset(p, 0, sizeof(float) * NX * NY);
    memset(visc, 0, sizeof(float) * NX * NY);
    memset(shear_rate, 0, sizeof(float) * NX * NY);
    memset(u_star, 0, sizeof(float) * (NX+1) * NY);
    memset(v_star, 0, sizeof(float) * NX * (NY+1));
    memset(p_prime, 0, sizeof(float) * NX * NY);
    memset(divergence, 0, sizeof(float) * NX * NY);
    memset(u_old, 0, sizeof(float) * (NX+1) * NY);
    memset(v_old, 0, sizeof(float) * NX * (NY+1));
    
    // Initialize flow field
    Initialize(u, v, p, visc, shear_rate, del_P, NX, NY);
    
    // Simulation parameters
    float physicalTime = 5.0f;        // Total physical time to simulate (seconds)
    float currentTime = 0.0f;         // Current physical time
    int timeStep = 0;                 // Current time step
    float cfl = 0.3f;                 // Target CFL number (REDUCED for stability)
    float maxResidual = 1.0f;         // Initial residual
    float convergenceTolerance = 1e-4f; // Convergence criteria (RELAXED for better convergence)
    int maxIterations = 50;           // Maximum SIMPLE iterations per time step (REDUCED)
    int outputInterval = 100;         // Output every N time steps
    
    // Ask user for simulation parameters
    printf("Enter total physical time (default: 5.0s): ");
    scanf("%f", &physicalTime);
    printf("Enter output interval in time steps (default: 100): ");
    scanf("%d", &outputInterval);
    
    // Calculate adaptive time step based on CFL condition
    float maxVelocity = U;  // Initial estimate
    float adaptiveDt = cfl * DX / maxVelocity;
    printf("Initial time step: %e seconds\n", adaptiveDt);
    
    // Create output file
    FILE *outfile = fopen("blood_flow_results.csv", "w");
    if (outfile == NULL) {
        fprintf(stderr, "Error: Could not open output file\n");
        exit(EXIT_FAILURE);
    }
    fprintf(outfile, "Time,TimeStep,Iterations,MaxResidual,MaxVelocity,MaxPressure\n");
    
    // Time advancement loop
    while (currentTime < physicalTime) {
        timeStep++;
        
        // Store previous time step values for temporal discretization
        memcpy(u_old, u, sizeof(float) * (NX+1) * NY);
        memcpy(v_old, v, sizeof(float) * NX * (NY+1));
        
        // SIMPLE algorithm iterations
        int iterations = 0;
        maxResidual = 1.0f;
        
        while (iterations < maxIterations && maxResidual > convergenceTolerance) {
            // Step 1: Apply boundary conditions
            ApplyBoundaryConditions(u, v, p, 0.0f, NX, NY);
            
            // Step 2: Momentum prediction
            Xmomentum_solver(u, v, p, visc, u_star, adaptiveDt, DX, DY, rho, NX, NY);
            YMomentum_solver(u, v, p, v_star, adaptiveDt, DX, DY, rho, visc, NX, NY);
            
            // Step 3: Calculate divergence
            Divergence(u_star, v_star, divergence, NX, NY);
            
            // Step 4: Solve pressure correction equation
            float omega = 1.5f;  // SOR relaxation parameter
            int converged = SolvePressurePoisson(divergence, p, p_prime, NX, NY, omega, convergenceTolerance/10.0f);
            
            // Step 5: Correct velocities and pressure
            CorrectVelocities(u, v, u_star, v_star, p, NX, NY);
            
            // Step 6: Update viscosity using Bird-Carreau model
            Shear_rate(u, v, visc, shear_rate, NX, NY);
            
            // Calculate residual for convergence check
            maxResidual = CalculateResidual(u, u_old, v, v_old, NX, NY);
            
            iterations++;
            
            // Store current values for next iteration
            memcpy(u_old, u, sizeof(float) * (NX+1) * NY);
            memcpy(v_old, v, sizeof(float) * NX * (NY+1));
        }
        
        // Update current time
        currentTime += adaptiveDt;
        
        // Calculate maximum velocity for adaptive time stepping
        float h_max_velocity = CalculateMaxVelocity(u, v, NX, NY);
        
        // Update time step based on CFL condition
        if (h_max_velocity > 0.01f) {  // Avoid division by very small values
            adaptiveDt = cfl * DX / h_max_velocity;
        }
        
        // Limit time step growth for stability
        adaptiveDt = fminf(adaptiveDt, 1.5f * adaptiveDt);
        
        // Adjust final time step to exactly reach physicalTime
        if (currentTime + adaptiveDt > physicalTime) {
            adaptiveDt = physicalTime - currentTime;
        }
        
        // Output results at specified intervals
        if (timeStep % outputInterval == 0 || currentTime >= physicalTime) {
            printf("Time: %.4f s, Step: %d, Iterations: %d, Residual: %.6e, Max Velocity: %.4f m/s\n", 
                   currentTime, timeStep, iterations, maxResidual, h_max_velocity);
            
            // Calculate maximum pressure
            float maxPressure = 0.0f;
            for (int i = 0; i < NX*NY; i++) {
                if (fabs(p[i]) > maxPressure) maxPressure = fabs(p[i]);
            }
            
            // Write to output file
            fprintf(outfile, "%.6f,%d,%d,%.6e,%.6f,%.6f\n", 
                    currentTime, timeStep, iterations, maxResidual, h_max_velocity, maxPressure);
            
            // Write field data to VTK file for visualization
            char filename[256];
            sprintf(filename, "blood_flow_t%.4f.vtk", currentTime);
            writeVTKFile(filename, u, v, p, NX, NY, DX, DY);
        }
    }
    
    // Stop timing and report
    clock_t end_time = clock();
    double elapsed_time = (double)(end_time - start_time) / CLOCKS_PER_SEC;
    printf("Total simulation time: %.2f seconds\n", elapsed_time);
    
    // Close output file
    fclose(outfile);
    
    // Free memory
    free(u);
    free(v);
    free(p);
    free(visc);
    free(shear_rate);
    free(u_star);
    free(v_star);
    free(p_prime);
    free(divergence);
    free(u_old);
    free(v_old);
    
    printf("\nSimulation completed successfully!\n");
    printf("To visualize results in ParaView:\n");
    printf("1. Open ParaView\n");
    printf("2. File -> Open -> Select blood_flow_t*.vtk files\n");
    printf("3. Click 'Apply' in the Properties panel\n");
    printf("4. Select visualization options (e.g., velocity_magnitude with Surface representation)\n");
    
    return 0;
}

