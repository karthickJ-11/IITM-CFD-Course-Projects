// Blood through arteries (LAD) artery analysis
// The flow is assumed to be unsteady, incompressible, laminar, non-Newtonian, and fully developed

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
#define V  0.0f       // velocity of the blood in m/s in y direction in the artery
#define DX 0.0001f    // length of the cell in m for x direction (100micrometers)
#define DY 0.0001f    // length of the cell in m for y direction (100micrometers)
#define dt 1.0e-5f    // time step in seconds
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

// Forward declarations
__global__ void Initialize(float *u, float *v, float *p, float *visc, float *shear_rate, float p_init, int nx, int ny);
__global__ void ApplyBoundaryConditions(float *u, float *v, float *p, float p_out, int nx, int ny);
__global__ void Shear_rate(float *u, float *v, float *mu, float *shear_rate, int nx, int ny);
__global__ void Xmomentum_solver(float *u_x, float *u_y, float *p, float *mu, float *u_x_new, float dt, float DX, float DY, float rho, int 1401, int 36);
__global__ void YMomentum_solver(float* u_x, float* u_y, float* p, float* v_new, float dt, float DX, float DY, float rho, float* mu, int 1401, int 36);
__global__ void Divergence(float *u_x_new, float *v_new, float *divergence, int nx, int ny);
__global__ void PoissonSolverGS(float *divergence, float *p, float *p_new, int nx, int ny, float omega, float *residual);
__global__ void CorrectVelocities(float *u, float *v, float *u_star, float *v_star, float *p, int nx, int ny);
__global__ void CalculateResidual(float *d_u, float *d_u_old, float *d_v, float *d_v_old, float *residual, int nx, int ny);
__global__ void CalculateMaxVelocity(float *d_u, float *d_v, float *max_velocity, int nx, int ny);
bool SolvePressurePoisson(float *d_divergence, float *d_p, float *d_p_new, int nx, int ny, float omega, float tolerance);
void writeVTKFile(const char* filename, float *u, float *v, float *p, int nx, int ny, float dx, float dy);

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
            u[idx_u] = u[idx_u-1];  // Zero gradient
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
__global__ void ApplyBoundaryConditions(float *u, float *v, float *p, float p_out, int nx, int ny) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    
    // Inlet boundary (west face, x = 0)
    if (i == 0 && j < ny) {
        // u-velocity at inlet face (i=0)
        float y_norm = 2.0f * (float)j/(ny-1) - 1.0f;
        u[j*(nx+1) + i] = U * (1.0f - y_norm*y_norm);  // Parabolic profile
        
        // v-velocity near inlet
        if (j < ny) {
            v[j*nx + i] = 0.0f;  // No y-component at inlet
        }
        
        // Pressure at inlet cells
        p[j*nx + i] = p[j*nx + (i+1)];  // Zero gradient
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
        
        // u-velocity near bottom wall
        if (i < nx+1) {
            u[j*(nx+1) + i] = 0.0f;  // No-slip condition
        }
    }
    
    // Wall boundaries (north face, y = ny)
    if (j == ny && i < nx) {
        // v-velocity at top wall (j=ny)
        v[j*nx + i] = 0.0f;  // No-slip condition
        
        // u-velocity near top wall
        if (i < nx+1) {
            u[(j-1)*(nx+1) + i] = 0.0f;  // No-slip condition
        }
    }
}

// Calculate shear rate and update viscosity using Bird-Carreau model
__global__ void Shear_rate(float *u, float *v, float *mu, float *shear_rate, int nx, int ny) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (i > 0 && i < nx-1 && j > 0 && j < ny-1) {
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
        
        // Store at cell center
        int idx = j*nx + i;
        shear_rate[idx] = sr;
        
        // Update viscosity using Bird-Carreau model
        mu[idx] = mu_inf + (mu0 - mu_inf) * powf(1.0f + powf(lambda * sr, a), (n-1.0f)/a);
    } else if (i < nx && j < ny) {
        // Set boundary values to zero or a safe default
        int idx = j*nx + i;
        shear_rate[idx] = 0.0f;
        mu[idx] = mu0;
    }
}

// X-momentum equation solver 
__global__ void Xmomentum_solver(float *u_x, float *u_y, float *p, float *mu, float *u_x_new, float dt, float DX, float DY, float rho, int nx, int ny) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (i >= 1 && i < nx-1 && j >= 1 && j < ny-1) {     
        // For staggered grid, u_x is stored at (i+1/2, j)
        int idx_u = j*(nx+1) + i;
        
        // 1. Calculate convection term using QUICK scheme
        // Calculate mass fluxes through cell faces
        float Fe = 0.5f * (u_x[j*(nx+1) + i] + u_x[j*(nx+1) + (i+1)]) * DY;
        float Fw = 0.5f * (u_x[j*(nx+1) + (i-1)] + u_x[j*(nx+1) + i]) * DY;
        
        // Interpolate v-velocities to u-cell faces
        float vn = 0.25f * (u_y[(j+1)*nx + i] + u_y[(j+1)*nx + (i-1)] + 
                           u_y[j*nx + i] + u_y[j*nx + (i-1)]);
        float vs = 0.25f * (u_y[j*nx + i] + u_y[j*nx + (i-1)] + 
                           u_y[(j-1)*nx + i] + u_y[(j-1)*nx + (i-1)]);
        
        float Fn = vn * DX;
        float Fs = vs * DX;
        
        // QUICK scheme for face values
        float phi_e, phi_w, phi_n, phi_s;
        
        // East face
        if (Fe >= 0) {
            phi_e = (3.0f/8.0f) * u_x[j*(nx+1) + (i+1)] + 
                    (6.0f/8.0f) * u_x[j*(nx+1) + i] - 
                    (1.0f/8.0f) * u_x[j*(nx+1) + (i-1)];
        } else {
            phi_e = (3.0f/8.0f) * u_x[j*(nx+1) + i] + 
                    (6.0f/8.0f) * u_x[j*(nx+1) + (i+1)] - 
                    (1.0f/8.0f) * u_x[j*(nx+1) + (i+2)];
        }
        
        // West face
        if (Fw >= 0) {
            phi_w = (3.0f/8.0f) * u_x[j*(nx+1) + i] + 
                    (6.0f/8.0f) * u_x[j*(nx+1) + (i-1)] - 
                    (1.0f/8.0f) * u_x[j*(nx+1) + (i-2)];
        } else {
            phi_w = (3.0f/8.0f) * u_x[j*(nx+1) + (i-1)] + 
                    (6.0f/8.0f) * u_x[j*(nx+1) + i] - 
                    (1.0f/8.0f) * u_x[j*(nx+1) + (i+1)];
        }
        
        // North face
        if (Fn >= 0) {
            phi_n = (3.0f/8.0f) * u_x[(j+1)*(nx+1) + i] + 
                    (6.0f/8.0f) * u_x[j*(nx+1) + i] - 
                    (1.0f/8.0f) * u_x[(j-1)*(nx+1) + i];
        } else {
            phi_n = (3.0f/8.0f) * u_x[j*(nx+1) + i] + 
                    (6.0f/8.0f) * u_x[(j+1)*(nx+1) + i] - 
                    (1.0f/8.0f) * u_x[(j+2)*(nx+1) + i];
        }
        
        // South face
        if (Fs >= 0) {
            phi_s = (3.0f/8.0f) * u_x[j*(nx+1) + i] + 
                    (6.0f/8.0f) * u_x[(j-1)*(nx+1) + i] - 
                    (1.0f/8.0f) * u_x[(j-2)*(nx+1) + i];
        } else {
            phi_s = (3.0f/8.0f) * u_x[(j-1)*(nx+1) + i] + 
                    (6.0f/8.0f) * u_x[j*(nx+1) + i] - 
                    (1.0f/8.0f) * u_x[(j+1)*(nx+1) + i];
        }
        
        float convection_term = -(Fe * phi_e - Fw * phi_w + Fn * phi_n - Fs * phi_s) / (DX * DY);
        
        // 2. Calculate diffusion term using central difference
        // Interpolate viscosity to u-cell faces
        float mu_e = 0.5f * (mu[j*nx + i] + mu[j*nx + (i+1)]);
        float mu_w = 0.5f * (mu[j*nx + (i-1)] + mu[j*nx + i]);
        float mu_n = 0.5f * (mu[j*nx + i] + mu[(j+1)*nx + i]);
        float mu_s = 0.5f * (mu[(j-1)*nx + i] + mu[j*nx + i]);

        // Calculate diffusion coefficients
        float De = mu_e * DY/DX;
        float Dw = mu_w * DY/DX;
        float Dn = mu_n * DX/DY;
        float Ds = mu_s * DX/DY;
        
        // Calculate diffusion term using coefficients
        float diffusion = De * (u_x[j*(nx+1) + (i+1)] - u_x[j*(nx+1) + i]) - 
                          Dw * (u_x[j*(nx+1) + i] - u_x[j*(nx+1) + (i-1)]) + 
                          Dn * (u_x[(j+1)*(nx+1) + i] - u_x[j*(nx+1) + i]) - 
                          Ds * (u_x[j*(nx+1) + i] - u_x[(j-1)*(nx+1) + i]);
        
        // 3. Calculate pressure gradient (staggered grid - pressure at cell centers)
        float pressure_gradient_term = (p[j*nx + i] - p[j*nx + (i-1)]) / DX;
        
        // 4. Update u_x using the momentum equation
        u_x_new[idx_u] = u_x[idx_u] + dt * (convection_term + diffusion/(DX*DY*rho) - pressure_gradient_term/rho);
    }
}

// Y-momentum equation solver
__global__ void YMomentum_solver(float* u_x, float* u_y, float* p, float* v_new, float dt, float DX, float DY, float rho, float* mu, int nx, int ny) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= 1 && i < nx-1 && j >= 1 && j < ny-1) {
        // For staggered grid, v is at (i,j+1/2)
        int idx = i + j*nx;
        
        // Interpolate u-velocities to v-cell faces (east and west)
        float ue = 0.25f * (u_x[i + j*(nx+1)] + u_x[(i+1) + j*(nx+1)] + 
                           u_x[i + (j-1)*(nx+1)] + u_x[(i+1) + (j-1)*(nx+1)]);
        float Fe = ue * DY;
        float uw = 0.25f * (u_x[(i-1) + j*(nx+1)] + u_x[i + j*(nx+1)] + 
                           u_x[(i-1) + (j-1)*(nx+1)] + u_x[i + (j-1)*(nx+1)]);
        float Fw = uw * DY;
        
        // Interpolate v-velocities to v-cell faces (north and south)
        float vn = 0.5f * (u_y[i + j*nx] + u_y[i + (j+1)*nx]);
        float Fn = vn * DX;
        float vs = 0.5f * (u_y[i + (j-1)*nx] + u_y[i + j*nx]);
        float Fs = vs * DX;
        
        // QUICK scheme for face values with flow direction check
        float phi_e, phi_w, phi_n, phi_s;
        
        // East face
        if (Fe >= 0) {
            phi_e = (3.0f/8.0f) * u_y[(i+1) + j*nx] + 
                    (6.0f/8.0f) * u_y[i + j*nx] - 
                    (1.0f/8.0f) * u_y[(i-1) + j*nx];
        } else {
            phi_e = (3.0f/8.0f) * u_y[i + j*nx] + 
                    (6.0f/8.0f) * u_y[(i+1) + j*nx] - 
                    (1.0f/8.0f) * u_y[(i+2) + j*nx];
        }
        
        // West face
        if (Fw >= 0) {
            phi_w = (3.0f/8.0f) * u_y[i + j*nx] + 
                    (6.0f/8.0f) * u_y[(i-1) + j*nx] - 
                    (1.0f/8.0f) * u_y[(i-2) + j*nx];
        } else {
            phi_w = (3.0f/8.0f) * u_y[(i-1) + j*nx] + 
                    (6.0f/8.0f) * u_y[i + j*nx] - 
                    (1.0f/8.0f) * u_y[(i+1) + j*nx];
        }
        
        // North face
        if (Fn >= 0) {
            phi_n = (3.0f/8.0f) * u_y[i + (j+1)*nx] + 
                    (6.0f/8.0f) * u_y[i + j*nx] - 
                    (1.0f/8.0f) * u_y[i + (j-1)*nx];
        } else {
            phi_n = (3.0f/8.0f) * u_y[i + j*nx] + 
                    (6.0f/8.0f) * u_y[i + (j+1)*nx] - 
                    (1.0f/8.0f) * u_y[i + (j+2)*nx];
        }
        
        // South face
        if (Fs >= 0) {
            phi_s = (3.0f/8.0f) * u_y[i + j*nx] + 
                    (6.0f/8.0f) * u_y[i + (j-1)*nx] - 
                    (1.0f/8.0f) * u_y[i + (j-2)*nx];
        } else {
            phi_s = (3.0f/8.0f) * u_y[i + (j-1)*nx] + 
                    (6.0f/8.0f) * u_y[i + j*nx] - 
                    (1.0f/8.0f) * u_y[i + (j+1)*nx];
        }
        
        // Viscosity interpolation - corrected indexing for staggered grid
        float mu_e = 0.5f * (mu[i + j*nx] + mu[(i+1) + j*nx]);
        float mu_w = 0.5f * (mu[(i-1) + j*nx] + mu[i + j*nx]);
        float mu_n = 0.5f * (mu[i + j*nx] + mu[i + (j+1)*nx]);
        float mu_s = 0.5f * (mu[i + (j-1)*nx] + mu[i + j*nx]);
        
        // Calculate diffusion coefficients
        float De = mu_e * DY/DX;
        float Dw = mu_w * DY/DX;
        float Dn = mu_n * DX/DY;
        float Ds = mu_s * DX/DY;
        
        // Calculate convection term
        float convection_term = -(Fe * phi_e - Fw * phi_w + Fn * phi_n - Fs * phi_s) / (DX * DY);
        
        // Calculate diffusion term using coefficients
        float diffusion = De * (u_y[(i+1) + j*nx] - u_y[i + j*nx]) - 
                          Dw * (u_y[i + j*nx] - u_y[(i-1) + j*nx]) + 
                          Dn * (u_y[i + (j+1)*nx] - u_y[i + j*nx]) - 
                          Ds * (u_y[i + j*nx] - u_y[i + (j-1)*nx]);
        
        // Pressure gradient - corrected for v-velocity at (i,j+1/2)
        float pressure_gradient_term = (p[i + j*nx] - p[i + (j-1)*nx]) / DY;
        
        // Update v-velocity
        v_new[idx] = u_y[idx] + dt * (convection_term + diffusion/(DX*DY*rho) - pressure_gradient_term/rho);
    }
}

// Calculate divergence for pressure correction
__global__ void Divergence(float *u_x_new, float *v_new, float *divergence, int nx, int ny) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (i >= 1 && i < nx-1 && j >= 1 && j < ny-1) {
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

// Pressure Poisson equation solver using Gauss-Seidel with SOR
__global__ void PoissonSolverGS(float *divergence, float *p, float *p_new, int nx, int ny, float omega, float *residual) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (i >= 1 && i < nx-1 && j >= 1 && j < ny-1) {
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
        
        // Calculate local residual for convergence check
        atomicAdd(residual, fabsf(p_new[idx] - p_old));
    }
}

// Host-side wrapper for Gauss-Seidel iteration with convergence check
bool SolvePressurePoisson(float *d_divergence, float *d_p, float *d_p_new, int nx, int ny, float omega, float tolerance) {
    dim3 block(16, 16);
    dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);
    
    float *d_residual;
    cudaMalloc((void**)&d_residual, sizeof(float));
    float h_residual;
    
    int max_iter = 1000;  // Maximum iterations
    
    for (int iter = 0; iter < max_iter; ++iter) {
        // Reset residual
        cudaMemset(d_residual, 0, sizeof(float));
        
        // Copy current pressure to p_new as initial guess
        cudaMemcpy(d_p_new, d_p, nx * ny * sizeof(float), cudaMemcpyDeviceToDevice);
        
        // Perform one Gauss-Seidel iteration
        PoissonSolverGS<<<grid, block>>>(d_divergence, d_p, d_p_new, nx, ny, omega, d_residual);
        cudaDeviceSynchronize();
        
        // Check convergence
        cudaMemcpy(&h_residual, d_residual, sizeof(float), cudaMemcpyDeviceToHost);
        h_residual /= (nx * ny);  // Normalize by grid size
        
        if (h_residual < tolerance) {
            cudaFree(d_residual);
            return true;  // Converged
        }
        
        // Update pressure for next iteration
        cudaMemcpy(d_p, d_p_new, nx * ny * sizeof(float), cudaMemcpyDeviceToDevice);
    }
    
    cudaFree(d_residual);
    return false;  // Did not converge within max_iter
}

// Correct velocities using pressure correction
__global__ void CorrectVelocities(float *u, float *v, float *u_star, float *v_star, float *p, int nx, int ny) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    
    // Correct u-velocity (staggered at i+1/2, j)
    if (i >= 1 && i < nx && j >= 1 && j < ny-1) {
        // d-coefficient from momentum equation (simplified)
        float d_u = dt / rho;
        
        // Proper indexing for staggered grid
        int idx_u = j*(nx+1) + i;  // u at (i+1/2,j)
        
        // Pressure gradient between adjacent cell centers
        float dp_dx = (p[j*nx + i] - p[j*nx + (i-1)]) / DX;
        
        // Correct u-velocity using pressure gradient
        u[idx_u] = u_star[idx_u] - d_u * dp_dx;
    }
    
    // Correct v-velocity (staggered at i, j+1/2)
    if (i >= 1 && i < nx-1 && j >= 1 && j < ny) {
        // d-coefficient from momentum equation (simplified)
        float d_v = dt / rho;
        
        // Proper indexing for staggered grid
        int idx_v = j*nx + i;  // v at (i,j+1/2)
        
        // Pressure gradient between adjacent cell centers
        float dp_dy = (p[j*nx + i] - p[(j-1)*nx + i]) / DY;
        
        // Correct v-velocity using pressure gradient
        v[idx_v] = v_star[idx_v] - d_v * dp_dy;
    }
}

// Calculate residual between iterations
__global__ void CalculateResidual(float *d_u, float *d_u_old, float *d_v, float *d_v_old, float *residual, int nx, int ny) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (i < nx && j < ny) {
        // Calculate u-velocity residual
        if (i < nx+1) {
            float u_diff = d_u[j*(nx+1) + i] - d_u_old[j*(nx+1) + i];
            atomicAdd(residual, u_diff*u_diff);
        }
        
        // Calculate v-velocity residual
        if (j < ny+1) {
            float v_diff = d_v[j*nx + i] - d_v_old[j*nx + i];
            atomicAdd(residual, v_diff*v_diff);
        }
    }
}

// Calculate maximum velocity for CFL condition
__global__ void CalculateMaxVelocity(float *d_u, float *d_v, float *max_velocity, int nx, int ny) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (i < nx && j < ny) {
        // Calculate velocity magnitude at cell center
        float u_center = 0.5f * (d_u[j*(nx+1) + i] + d_u[j*(nx+1) + (i+1)]);
        float v_center = 0.5f * (d_v[j*nx + i] + d_v[(j+1)*nx + i]);
        float velocity_mag = sqrtf(u_center*u_center + v_center*v_center);
        
        // Update maximum velocity using atomic operation
        atomicMax((int*)max_velocity, __float_as_int(velocity_mag));
    }
}

// Write VTK file for visualization
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
    // Initialize variables and parameters
    float *h_u, *h_v, *h_p, *h_visc, *h_shear_rate;
    float *d_u, *d_v, *d_p, *d_visc, *d_shear_rate;
    float *d_u_star, *d_v_star, *d_p_prime, *d_divergence;
    float *d_u_old, *d_v_old;
    
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
    
    // Set execution configuration
    dim3 blockSize(16, 16);
    dim3 gridSize((NX + blockSize.x - 1) / blockSize.x, 
                  (NY + blockSize.y - 1) / blockSize.y);
    
    // Initialize flow field on device
    Initialize<<<gridSize, blockSize>>>(d_u, d_v, d_p, d_visc, d_shear_rate, del_P, NX, NY);
    cudaDeviceSynchronize();
    
    // Simulation parameters
    float physicalTime = 5.0f;        // Total physical time to simulate (seconds)
    float currentTime = 0.0f;         // Current physical time
    int timeStep = 0;                 // Current time step
    float cfl = 0.5f;                 // Target CFL number
    float maxResidual = 1.0f;         // Initial residual
    float convergenceTolerance = 1e-5f; // Convergence criteria
    int maxIterations = 1000;         // Maximum SIMPLE iterations per time step
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
    fprintf(outfile, "Time,TimeStep,Iterations,MaxResidual,MaxVelocity,MaxPressure\n");
    
    // Start timing
    cudaEventRecord(start, 0);
    
    // Time advancement loop
    while (currentTime < physicalTime) {
        timeStep++;
        
        // Store previous time step values for temporal discretization
        cudaMemcpy(d_u_old, d_u, sizeof(float) * (NX+1) * NY, cudaMemcpyDeviceToDevice);
        cudaMemcpy(d_v_old, d_v, sizeof(float) * NX * (NY+1), cudaMemcpyDeviceToDevice);
        
        // SIMPLE algorithm iterations
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
            float omega = 1.5f;  // SOR relaxation parameter
            bool converged = SolvePressurePoisson(d_divergence, d_p, d_p_prime, NX, NY, omega, convergenceTolerance/10.0f);
            
            // Step 5: Correct velocities and pressure
            CorrectVelocities<<<gridSize, blockSize>>>(d_u, d_v, d_u_star, d_v_star, d_p, NX, NY);
            
            // Step 6: Update viscosity using Bird-Carreau model
            Shear_rate<<<gridSize, blockSize>>>(d_u, d_v, d_visc, d_shear_rate, NX, NY);
            
            // Calculate residual for convergence check
            float *d_residual;
            cudaMalloc((void**)&d_residual, sizeof(float));
            cudaMemset(d_residual, 0, sizeof(float));
            
            // Calculate velocity residual (change between iterations)
            CalculateResidual<<<gridSize, blockSize>>>(d_u, d_u_old, d_v, d_v_old, d_residual, NX, NY);
            cudaMemcpy(&maxResidual, d_residual, sizeof(float), cudaMemcpyDeviceToHost);
            
            cudaFree(d_residual);
            iterations++;
            
            // Store current values for next iteration
            cudaMemcpy(d_u_old, d_u, sizeof(float) * (NX+1) * NY, cudaMemcpyDeviceToDevice);
            cudaMemcpy(d_v_old, d_v, sizeof(float) * NX * (NY+1), cudaMemcpyDeviceToDevice);
        }
        
        // Update current time
        currentTime += adaptiveDt;
        
        // Calculate maximum velocity for adaptive time stepping
        float *d_max_velocity;
        cudaMalloc((void**)&d_max_velocity, sizeof(float));
        cudaMemset(d_max_velocity, 0, sizeof(float));
        
        CalculateMaxVelocity<<<gridSize, blockSize>>>(d_u, d_v, d_max_velocity, NX, NY);
        float h_max_velocity;
        cudaMemcpy(&h_max_velocity, d_max_velocity, sizeof(float), cudaMemcpyDeviceToHost);
        cudaFree(d_max_velocity);
        
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
            
            // Copy data back to host for output
            cudaMemcpy(h_u, d_u, sizeof(float) * (NX+1) * NY, cudaMemcpyDeviceToHost);
            cudaMemcpy(h_v, d_v, sizeof(float) * NX * (NY+1), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_p, d_p, sizeof(float) * NX * NY, cudaMemcpyDeviceToHost);
            cudaMemcpy(h_visc, d_visc, sizeof(float) * NX * NY, cudaMemcpyDeviceToHost);
            cudaMemcpy(h_shear_rate, d_shear_rate, sizeof(float) * NX * NY, cudaMemcpyDeviceToHost);
            
            // Calculate maximum pressure
            float maxPressure = 0.0f;
            for (int i = 0; i < NX*NY; i++) {
                if (fabs(h_p[i]) > maxPressure) maxPressure = fabs(h_p[i]);
            }
            
            // Write to output file
            fprintf(outfile, "%.6f,%d,%d,%.6e,%.6f,%.6f\n", 
                    currentTime, timeStep, iterations, maxResidual, h_max_velocity, maxPressure);
            
            // Write field data to VTK file for visualization
            char filename[256];
            sprintf(filename, "blood_flow_t%.4f.vtk", currentTime);
            writeVTKFile(filename, h_u, h_v, h_p, NX, NY, DX, DY);
        }
    }
    
    // Stop timing and report
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("Total simulation time: %.2f seconds\n", elapsedTime/1000.0f);
    
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
    
    return 0;
}

