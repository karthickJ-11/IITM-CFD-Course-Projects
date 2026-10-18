#include <stdio.h>
#include <math.h>
#include <time.h>

#define M 41 // Number of cells in Y axis (height)
#define N 21 // Number of cells in X axis (width)
#define T_BOTTOM 100.0 // Bottom boundary temperature
#define T_OTHER 30.0 // Other boundaries temperature
#define ERROR_THRESHOLD 0.01 // Convergence criteria
#define K 50.0 // Thermal conductivity (W/mK)

double T[M][N]; // Temperature grid

void initializeGrid() {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            T[i][j] = 0; // Set initial temperature to zero
        }
    }
}

void setBoundaryConditions() {
    for (int j = 0; j < N; j++) {
        T[M-1][j] = T_BOTTOM; // Bottom boundary temperature
    }
    for (int j = 0; j < N; j++) {
        T[0][j] = T_OTHER; // Top boundary temperature
    }
    for (int i = 0; i < M; i++) {
        T[i][0] = T_OTHER; // Left boundary temperature
    }
    for (int i = 0; i < M; i++) {
        T[i][N-1] = T_OTHER; // Right boundary temperature
    }
}

void printGrid(const char* title, FILE *file) {
    fprintf(file, "%s:\n", title);
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            fprintf(file, "%5.1f ", T[i][j]);
        }
        fprintf(file, "\n");
    }
}

double calculateHeatTransfer() {
    double total_heat_transfer = 0.0;
    double dx = 1.0 / (N - 1); // Assuming unit length in X direction
    for (int j = 0; j < N - 1; j++) {
        total_heat_transfer += (T[M-1][j] + T[M-1][j+1]) / 2 * dx;
    }
    return total_heat_transfer * K;
}

int main() {
    FILE *outputFile;
    outputFile = fopen("temperature_data.txt", "w"); // Open file for writing

    if (outputFile == NULL) {
        printf("Error opening file!\n");
        return 1;
    }

    initializeGrid();
    setBoundaryConditions();

    printGrid("Initial Temperature Grid", outputFile);

    double max_error;
    clock_t start_time, end_time;

    int iteration_count = 0;

    do {
        start_time = clock(); // Start timer
        max_error = 0.0;

        // Sweep along the x-axis 
        for (int i = 1; i < M - 1; i++) {
            for (int j = 1; j < N - 1; j++) {
                double old_temp = T[i][j];
                T[i][j] = 0.25 * (T[i + 1][j] + T[i - 1][j] + T[i][j + 1] + T[i][j - 1]);

                double error = fabs(T[i][j] - old_temp);
                if (error > max_error) {
                    max_error = error;
                }
            }
        }

        // Sweep along the y-axis 
        for (int j=1; j < N - 1; j++) {
            for (int i=1; i < M - 1; i++) {
                double old_temp = T[i][j];
                T[i][j] = 0.25 * (T[i + 1][j] + T[i - 1][j] + T[i][j + 1] + T[i][j - 1]);

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

    } while (max_error > ERROR_THRESHOLD); // Continue until convergence

    printGrid("Final Temperature Grid", outputFile);

    double heatTransfer = calculateHeatTransfer();
    printf("Total Heat Transfer along Bottom Boundary: %.2f W\n", heatTransfer);
    
    fprintf(outputFile, "Total Heat Transfer along Bottom Boundary: %.2f W\n", heatTransfer);

    fclose(outputFile); // Close the file

    return 0;
}
