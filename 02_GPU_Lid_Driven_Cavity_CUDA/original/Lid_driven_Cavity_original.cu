#include <stdio.h>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <sys/stat.h>

// Simulation Parameters
#define L 1.0
#define DX 0.0025
#define DY 0.0025
#define Re 100.0
#define Dt 0.001
#define MaxIter 50000
#define Error 1e-5
#define U_Wall 1.0
#define NX ((int)(L/DX) + 1)
#define NY ((int)(L/DY) + 1)
#define OUTPUT_INTERVAL 1

// CUDA Error Checking
#define CHECK(call) { \
    cudaError_t err = call; \
    if(err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error: %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
}

// Kernel Prototypes with C linkage
#ifdef __cplusplus
extern "C" {
#endif
__global__ void apply_boundary_conditions(double *Vort, double *Str);
__global__ void update_vorticity(double *Vort, double *Vop, double *Str);
__global__ void solve_stream_function(double *Str, double *Vort, int phase);
__global__ void calculate_velocity(double *u, double *v, double *Str);
#ifdef __cplusplus
}
#endif

// ParaView-compatible Tecplot output
void write_tecplot2D(double* u, double* v, double* vort, double* stream, 
                    int nx, int ny, double dx, double dy, int iter) {
    char filename[256];
    FILE *fp;
    
    // Create output directory
    struct stat st = {0};
    if (stat("output", &st) == -1) mkdir("output", 0700);

    sprintf(filename, "output/flow_%04d.dat", iter);
    fp = fopen(filename, "w");

    // Write header
    fprintf(fp, "TITLE = \"Lid-Driven Cavity Flow\"\n");
    fprintf(fp, "VARIABLES = \"X\", \"Y\", \"U\", \"V\", \"Vorticity\", \"Stream\"\n");
    fprintf(fp, "ZONE I=%d, J=%d, DATAPACKING=POINT\n", nx, ny);

    // Write data with fixed-width format
    for (int j = 0; j < ny; j++) {
        for (int i = 0; i < nx; i++) {
            double x = i * dx;
            double y = j * dy;
            int idx = j * nx + i;
            fprintf(fp, "%10.6e %10.6e %10.6e %10.6e %10.6e %10.6e\n", 
                    x, y, 
                    u[idx], v[idx], 
                    vort[idx], stream[idx]);
        }
    }
    fclose(fp);
}

int main() {
    // Allocate Unified Memory
    double *Vort, *Vop, *Str, *u, *v;
    CHECK(cudaMallocManaged(&Vort, NX*NY*sizeof(double)));
    CHECK(cudaMallocManaged(&Vop, NX*NY*sizeof(double)));
    CHECK(cudaMallocManaged(&Str, NX*NY*sizeof(double)));
    CHECK(cudaMallocManaged(&u, NX*NY*sizeof(double)));
    CHECK(cudaMallocManaged(&v, NX*NY*sizeof(double)));

    // Initialize Arrays
    for(int i=0; i<NX*NY; i++) {
        Vort[i] = 0.0; Vop[i] = 0.0; Str[i] = 0.0; u[i] = 0.0; v[i] = 0.0;
    }

    // CUDA Grid Configuration
    dim3 block(16, 16);
    dim3 grid((NX + block.x - 1)/block.x, (NY + block.y - 1)/block.y);

    // Main Simulation Loop
    int converged = 0;
    for(int iter=0; iter<MaxIter && !converged; iter++) {
        // Apply Boundary Conditions first
        apply_boundary_conditions<<<grid, block>>>(Vort, Str);
        CHECK(cudaDeviceSynchronize());

        // Store previous vorticity (with BCs applied)
        CHECK(cudaMemcpy(Vop, Vort, NX*NY*sizeof(double), cudaMemcpyDeviceToDevice));

        // Update Vorticity using previous state (Vop)
        update_vorticity<<<grid, block>>>(Vort, Vop, Str);
        CHECK(cudaDeviceSynchronize());

        // Solve Stream Function (Red-Black Gauss-Seidel)
        for(int phase=0; phase<2; phase++) {
            solve_stream_function<<<grid, block>>>(Str, Vort, phase);
            CHECK(cudaDeviceSynchronize());
        }

        // Check convergence every iteration after 10 steps
        if(iter > 1) {
            double max_error = 0.0;
            for(int idx=0; idx<NX*NY; idx++) {
                double error = fabs(Vort[idx] - Vop[idx]);
                if(error > max_error) max_error = error;
            }
            printf("Iter %d: Max Error = %e\n", iter, max_error);
            if(max_error < Error) {
                converged = 1;
                printf("Converged after %d iterations\n", iter);
            }
        }

        // Output every OUTPUT_INTERVAL steps
        if(iter % OUTPUT_INTERVAL == 0) {
            calculate_velocity<<<grid, block>>>(u, v, Str);
            CHECK(cudaDeviceSynchronize());
            write_tecplot2D(u, v, Vort, Str, NX, NY, DX, DY, iter);
        }
    }

    // Final Output
    calculate_velocity<<<grid, block>>>(u, v, Str);
    CHECK(cudaDeviceSynchronize());
    write_tecplot2D(u, v, Vort, Str, NX, NY, DX, DY, MaxIter);

    // Free Memory
    CHECK(cudaFree(Vort));
    CHECK(cudaFree(Vop));
    CHECK(cudaFree(Str));
    CHECK(cudaFree(u));
    CHECK(cudaFree(v));

    return 0;
}

// Boundary Conditions Kernel (equations 30-33 from PDF)
__global__ void apply_boundary_conditions(double *Vort, double *Str) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if(i >= NX || j >= NY) return;

    const double h2 = DX * DX;

    // Top Boundary (equation 30)
    if(j == NY-1 && i > 0 && i < NX-1) {
        Vort[i*NY + j] = (-2*Str[i*NY + (j-1)]/h2) - (2*U_Wall/DX);
    }
    // Bottom Boundary (equation 31)
    else if(j == 0 && i > 0 && i < NX-1) {
        Vort[i*NY + j] = -2*Str[i*NY + (j+1)]/h2;
    }
    // Left Boundary (equation 32)
    else if(i == 0 && j > 0 && j < NY-1) {
        Vort[i*NY + j] = -2*Str[(i+1)*NY + j]/h2;
    }
    // Right Boundary (equation 33)
    else if(i == NX-1 && j > 0 && j < NY-1) {
        Vort[i*NY + j] = -2*Str[(i-1)*NY + j]/h2;
    }
}

// Vorticity Update Kernel (equation 16 from PDF)
__global__ void update_vorticity(double *Vort, double *Vop, double *Str) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if(i <= 0 || i >= NX-1 || j <= 0 || j >= NY-1) return;

    const double inv2h = 1.0/(2.0*DX);
    const double h2 = DX*DX;

    // Calculate derivatives using previous state (Vop)
    double dStr_dy = (Str[i*NY + (j+1)] - Str[i*NY + (j-1)]) * inv2h;
    double dStr_dx = (Str[(i+1)*NY + j] - Str[(i-1)*NY + j]) * inv2h;
    double dVop_dx = (Vop[(i+1)*NY + j] - Vop[(i-1)*NY + j]) * inv2h;
    double dVop_dy = (Vop[i*NY + (j+1)] - Vop[i*NY + (j-1)]) * inv2h;

    // Calculate Laplacian (equation 16)
    double laplacian = (Vop[(i+1)*NY + j] + Vop[(i-1)*NY + j] +
                      Vop[i*NY + (j+1)] + Vop[i*NY + (j-1)] -
                      4*Vop[i*NY + j]) / h2;

    // Update vorticity (equation 16)
    Vort[i*NY + j] = Vop[i*NY + j] + Dt * (
        (laplacian/Re) - (dStr_dy*dVop_dx - dStr_dx*dVop_dy)
    );
}

// Stream Function Solver (equation 18 from PDF)
__global__ void solve_stream_function(double *Str, double *Vort, int phase) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if((i+j)%2 != phase) return;
    if(i <= 0 || i >= NX-1 || j <= 0 || j >= NY-1) return;

    Str[i*NY + j] = 0.25 * (
        Str[(i+1)*NY + j] + Str[(i-1)*NY + j] +
        Str[i*NY + (j+1)] + Str[i*NY + (j-1)] +
        DX*DX*Vort[i*NY + j]
    );
}

// Velocity Calculation (equations 34-35 from PDF)
__global__ void calculate_velocity(double *u, double *v, double *Str) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if(i <= 0 || i >= NX-1 || j <= 0 || j >= NY-1) return;

    const double inv2h = 1.0 / (2.0 * DX);
    
    // u = dψ/dy (equation 34)
    u[j*NX + i] = (Str[(j+1)*NX + i] - Str[(j-1)*NX + i]) * inv2h;
    
    // v = -dψ/dx (equation 35)
    v[j*NX + i] = -(Str[j*NX + (i+1)] - Str[j*NX + (i-1)]) * inv2h;
    
    // Apply top wall velocity
    if(j == NY-1 && i > 0 && i < NX-1) {
        u[j*NX + i] = U_Wall;
    }
}

