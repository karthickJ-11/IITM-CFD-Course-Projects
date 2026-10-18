// 2-D steady heat conduction, Laplace equation on a 1 x 2 plate, 21 x 41 nodes
// Scheme: ADI with relaxation (alternating x/y line SOR)
// Original author: Karthick J (OE23S018). Corrections marked [FIX]/[ADDED].
#include <stdio.h>
#include <math.h>
#include <time.h>

#define M 41  // Number of cells in Y axis (height)
#define N 21  // Number of cells in X axis (width)
#define T_BOTTOM 100.0 // Bottom boundary temperature
#define T_OTHER 30.0   // Other boundaries temperature
#define ERROR_THRESHOLD 0.01 // Convergence criteria
#define K 50.0 // Thermal conductivity (W/mK)

// Temperature grid
double T[M][N];

void initializeGrid() 
{
    for (int i = 0; i < M; i++) 
    {
        for (int j = 0; j < N; j++) 
        {
            T[i][j] = 0; // Set initial temperature to zero
        }
    }
}

// Set boundary conditions
void setBoundaryConditions() 
	{
    for (int j = 0; j < N; j++) 
    { // Bottom wall (X-axis)
        T[M-1][j] = T_BOTTOM; // Set bottom boundary temperature
    }
    for (int j = 0; j < N; j++) 
    { // Top wall (X-axis)
        T[0][j] = T_OTHER; // Set top boundary temperature
    }
    for (int i = 0; i < M; i++) 
    { // Left wall (Y-axis)
        T[i][0] = T_OTHER; // Set left boundary temperature
    }
    for (int i = 0; i < M; i++) 
    { // Right wall (Y-axis)
        T[i][N-1] = T_OTHER; // Set right boundary temperature
    }
}

// Print grid to file
void printGrid(const char* title, FILE *file) 
{
    fprintf(file, "%s:\n", title);
    for (int i = 0; i < M; i++) 
    {
        for (int j = 0; j < N; j++) 
        {
            fprintf(file, "%5.1f ", T[i][j]);
        }
        fprintf(file, "\n");
    }
}

// Calculate heat transfer along the bottom boundary
// [FIX: the original returned K * integral(T dx), i.e. conductivity x temperature, which is not a heat
//  rate. Fourier's law gives q = -K dT/dn. Here the heat entering through the bottom wall (per unit
//  depth) is summed over the bottom faces of the first interior row: Q = K * sum (T_wall - T_1)/dy * dx.
//  This is the discrete flux used by the 5-point stencil, so the energy balance closes exactly.]
double calculateHeatTransfer() 
{
    double total_heat_transfer = 0.0;
    double dx = 1.0 / (N - 1); // Assuming unit length in X direction
    double dy = 2.0 / (M - 1); // Plate height 2 units in Y direction
    for (int j = 1; j < N - 1; j++) 
    {
        total_heat_transfer += (T[M-1][j] - T[M-2][j]) / dy * dx;
    }
    return total_heat_transfer * K; // Multiply by thermal conductivity
}

// [ADDED for post-processing only: full-precision final field, one row of nodes per line]
void writeFinalField(const char* name) 
{
    FILE *f = fopen(name, "w");
    for (int i = 0; i < M; i++) 
    {
        for (int j = 0; j < N; j++) fprintf(f, "%.10e ", T[i][j]);
        fprintf(f, "\n");
    }
    fclose(f);
}

// [FIX: line methods need an implicit (tridiagonal) solve along each line. The original
//  "line" and "ADI" codes only repeated point-wise updates. Thomas algorithm (TDMA):
//  a[k] x[k-1] + b[k] x[k] + c[k] x[k+1] = d[k],  k = 0..n-1]
void tdma(int n, double *a, double *b, double *c, double *d, double *x) 
{
    double cp[256], dp[256];
    cp[0] = c[0] / b[0];
    dp[0] = d[0] / b[0];
    for (int k = 1; k < n; k++) 
    {
        double m = b[k] - a[k] * cp[k-1];
        cp[k] = c[k] / m;
        dp[k] = (d[k] - a[k] * dp[k-1]) / m;
    }
    x[n-1] = dp[n-1];
    for (int k = n - 2; k >= 0; k--) x[k] = dp[k] - cp[k] * x[k+1];
}

// Implicit solve along row i (x-direction), neighbours in y taken at their latest values.
// omega = 1 -> line Gauss-Seidel, omega > 1 -> line SOR. Returns the max change on the row.
double rowSweep(int i, double omega) 
{
    double a[256], b[256], c[256], d[256], x[256]; int n = N - 2; double err = 0.0;
    for (int k = 0; k < n; k++) 
    {
        int j = k + 1;
        a[k] = -1.0; b[k] = 4.0; c[k] = -1.0;
        d[k] = T[i-1][j] + T[i+1][j];
        if (j == 1)     d[k] += T[i][0];
        if (j == N - 2) d[k] += T[i][N-1];
    }
    a[0] = 0.0; c[n-1] = 0.0;
    tdma(n, a, b, c, d, x);
    for (int k = 0; k < n; k++) 
    {
        int j = k + 1; double old_temp = T[i][j];
        T[i][j] = (1 - omega) * old_temp + omega * x[k];
        if (fabs(T[i][j] - old_temp) > err) err = fabs(T[i][j] - old_temp);
    }
    return err;
}

// Implicit solve along column j (y-direction), neighbours in x taken at their latest values.
double colSweep(int j, double omega) 
{
    double a[256], b[256], c[256], d[256], x[256]; int n = M - 2; double err = 0.0;
    for (int k = 0; k < n; k++) 
    {
        int i = k + 1;
        a[k] = -1.0; b[k] = 4.0; c[k] = -1.0;
        d[k] = T[i][j-1] + T[i][j+1];
        if (i == 1)     d[k] += T[0][j];
        if (i == M - 2) d[k] += T[M-1][j];
    }
    a[0] = 0.0; c[n-1] = 0.0;
    tdma(n, a, b, c, d, x);
    for (int k = 0; k < n; k++) 
    {
        int i = k + 1; double old_temp = T[i][j];
        T[i][j] = (1 - omega) * old_temp + omega * x[k];
        if (fabs(T[i][j] - old_temp) > err) err = fabs(T[i][j] - old_temp);
    }
    return err;
}

int main() 
{
    FILE *outputFile;
    outputFile = fopen("temperature_data.txt", "w"); // Open file for writing

    if (outputFile == NULL) 
    {
        printf("Error opening file!\n");
        return 1;
    }

    // Initialize the grid and set boundary conditions
    initializeGrid();
    setBoundaryConditions();

    printGrid("Initial Temperature Grid", outputFile);

    double max_error;
    clock_t start_time, end_time;
    
    int iteration_count = 0; // Counter for iterations
    double omega = 1.5; // Relaxation factor

    do {
        start_time = clock(); // Start timer
        max_error = 0.0;

        // ADI with relaxation: both half-iterations over-relaxed with omega.
        // [FIX: the original update T = (1-w)*T_old + w*0.5*T_new converges to a wrong fixed
        //  point T = 0.5*T_new (the factor 0.5 "iteration level" scales the solution), which is
        //  why it stopped after 9 iterations with a non-physical field.]
        for (int i = M - 2; i >= 1; i--) 
        {
            double error = rowSweep(i, omega);
            if (error > max_error) max_error = error;
        }
        for (int j = 1; j < N - 1; j++) 
        {
            double error = colSweep(j, omega);
            if (error > max_error) max_error = error;
        }

        end_time = clock(); // End timer
        
        iteration_count++; // Increment iteration count

        fprintf(outputFile, "Iteration %d Time: %.6f seconds\n", 
                iteration_count, 
                (double)(end_time - start_time) / CLOCKS_PER_SEC);
                
        printGrid("Temperature Grid After Iteration", outputFile);

        printf("Iteration %d Max error: %.6f\n", iteration_count, max_error);

    }
     while (max_error > ERROR_THRESHOLD); // Continue until convergence

    printGrid("Final Temperature Grid", outputFile);
    writeFinalField("final_field.dat"); // [ADDED]

    double heatTransfer = calculateHeatTransfer();
    printf("Total Heat Transfer along Bottom Boundary: %.2f W\n", heatTransfer);
    
    fprintf(outputFile, "Total Heat Transfer along Bottom Boundary: %.2f W\n", heatTransfer);

    fclose(outputFile); // Close the file

    return 0;
}
