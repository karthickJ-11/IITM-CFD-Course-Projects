#include <stdio.h>
#include <math.h>
//**********************ADVECTION-DIFFUSION SIMULATION WITH CONTINUOUS SOURCE**********************//
// Original author: Karthick J (OE23S018), OE5450 Tutorial 5. Code taken from the submitted report.
// Corrections are marked [FIX]. Note: the dye is released ONCE at t = 0 (instantaneous point
// release); the update does not re-inject it, so the 'continuous source' wording refers to the
// initial condition only.
// Domain parameters
#define L 20.0     // Domain length (-10m to +10m)
#define DX 0.2     // Spatial step size (m)
#define Cd 10.0    // Initial concentration (kg/m³)
#define D 0.5      // Diffusion coefficient (m²/s)
#define DT 0.01    // Time step (s)
#define U 2.0      // Flow velocity (m/s)
#define T_MAX 100.0 // Total simulation time (s)
#define NX (int)(L/DX) + 1 // Number of spatial points
// Prototypes
void initialize(double C[], int center);
void update(double C[], double C_new[], double dx, double dt, double u, double
d, int center);
void save_results(FILE *fp, double C[], double time);
int main() {
  // Simulation parameters
  const int center = (NX-1)/2;
  const int num_steps = (int)(T_MAX/DT);
  double current_time = 0.0;

  // Concentration arrays
  // [FIX: NX is not an integer constant expression, so these are variable-length arrays,
  //  which C does not allow to be initialised with '= {0.0}' (compile error). Zero them in a loop.]
  double C_old[NX];
  double C_new[NX];
  for (int i = 0; i < NX; i++) { C_old[i] = 0.0; C_new[i] = 0.0; }

  // Stability checks
  const double CFL = U * DT / DX;
  const double diffusion_number = D * DT / (DX*DX);

  // File handling
  FILE *fp = fopen("results.txt", "w");
  if(!fp) {
      printf("Error opening output file!\n");
      return 1;
  }

  // Write simulation parameters
  fprintf(fp, "# Advection-Diffusion Simulation Parameters\n");
    fprintf(fp, "# Domain Length: %.1f m\n", L);
    fprintf(fp, "# Velocity: %.1f m/s\n", U);
    fprintf(fp, "# Diffusion Coefficient: %.1f m²/s\n", D);
    fprintf(fp, "# CFL Number: %.3f\n", CFL);
    fprintf(fp, "# Diffusion Number: %.3f\n\n", diffusion_number);
    // Initialization and save initial condition
    initialize(C_old, center);
    save_results(fp, C_old, 0.0); // Save initial condition at t=0
    // Main simulation loop
    for(int n = 0; n < num_steps; n++) {
        current_time = (n + 1) * DT;   // [FIX: missing semicolon]
        // Update concentrations
        update(C_old, C_new, DX, DT, U, D, center);

        // Save data at specified intervals
        if(fmod(current_time, 1.0) <= DT/2.0) {
             save_results(fp, C_old, current_time);
        }
    }
    fclose(fp);
    printf("Simulation completed successfully!\n");
    return 0;
}
void initialize(double C[], int center) {
    for(int i = 0; i < NX; i++) {
        C[i] = 0.0;
            // Continuous source at center
    }
    C[center] = Cd;
}
void update(double C[], double C_new[], double dx, double dt, double u, double
d, int center) {
    // Update interior points
    for(int i = 1; i < NX-1; i++) {
        const double adv = u * (C[i] - C[i-1]) / dx;
        const double diff = d * (C[i+1] - 2*C[i] + C[i-1]) / (dx*dx);
        C_new[i] = C[i] - dt * adv + dt * diff;
    }
    // Boundary conditions
    C_new[0] = 0.0;         // Upstream Dirichlet BC
    C_new[NX-1] = 0.0;          // Downstream Dirichlet BC
    // Update array
    for(int i = 0; i < NX; i++) {
        C[i] = C_new[i];
    }}
void save_results(FILE *fp, double C[], double time) {
    fprintf(fp, "# Time = %.2f s\n", time);
    fprintf(fp, "%-12s %-12s\n", "Position(m)", "Concentration");
    for(int i = 0; i < NX; i++) {
        double x = -L/2 + i * DX;
        fprintf(fp, "%10.2f %14.4f\n", x, C[i]);
    }
    fprintf(fp, "\n\n"); // Two blank lines for gnuplot index
}
