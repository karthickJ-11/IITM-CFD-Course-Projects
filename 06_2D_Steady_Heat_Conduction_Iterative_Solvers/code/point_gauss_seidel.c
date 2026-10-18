// 2-D steady heat conduction, Laplace equation on a 1 x 2 plate, 21 x 41 nodes
// Scheme: Point Gauss-Seidel
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

    do {
        start_time = clock(); // Start timer
        max_error = 0.0;

        for (int i = 1; i < M - 1; i++) 
        {
            for (int j = 1; j < N - 1; j++) 
            {
                double old_temp = T[i][j];
                T[i][j] = 0.25 * (T[i + 1][j] + T[i - 1][j] + T[i][j + 1] + T[i][j - 1]); 

                // Calculate maximum error
                double error = fabs(T[i][j] - old_temp);
                if (error > max_error) {
                    max_error = error;
                }
            }
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
