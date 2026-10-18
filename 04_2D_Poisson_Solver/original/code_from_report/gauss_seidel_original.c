#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define L 1.0 // length of the domain (x and y)
#define delta_x 0.01 // length of the cell size in x axis
#define delta_y 0.01 // length of the cell size in y axis
#define MX ((int)(L / delta_x) + 1) // number of cells in x axis
#define NY ((int)(L / delta_y) + 1) // number of cells in y axis
#define ERROR 0.0001 // error for the convergence
#define T_max 10.0 // maximum temperature at the boundary condition
#define T_min 0.0 // minimum temperature at the boundary condition
#define Max_iteration 10000 // maximum iteration allowed for solving the equation

double T[MX][NY]; // Temperature grid
double T_new[MX][NY]; // New temperature grid

// Function to set boundary conditions
void setBoundaryConditions() {
    int i, j;
    // Top side Boundary condition
    for (i = 0; i < MX; i++) {
        T[i][NY - 1] = T_max;
    }
    // Bottom side Boundary condition
    for (i = 0; i < MX; i++) {
        T[i][0] = T_max;
    }
    // Left side Boundary condition
    for (j = 0; j < NY; j++) {
        T[0][j] = T_min;
    }
    // Right side Boundary condition
    for (j = 0; j < NY; j++) {
        T[MX - 1][j] = T_min;
    }
}

// Function to perform Gauss-Seidel iteration
void gaussSeidelIteration(double *max_diff) {
    int i, j;
    double q1 = 1.0, q2 = -1.0;
    int sx1 = (int)(0.45 / delta_x), sy1 = (int)(0.5 / delta_y);
    int sx2 = (int)(0.55 / delta_x), sy2 = (int)(0.5 / delta_y);

    *max_diff = 0.0; // Reset max_diff

    for (i = 1; i < MX - 1; i++) {
        for (j = 1; j < NY - 1; j++) {
            double source = 0;
            if (i == sx1 && j == sy1) source = q1;
            if (i == sx2 && j == sy2) source = q2;

            // Update T[i][j] using the most recent values
            double old_T = T[i][j];
            T_new[i][j] = 0.25 * (T[i + 1][j] + T[i - 1][j] + T[i][j + 1] + T[i][j - 1] - (delta_x * delta_x) * source);

            // Calculate error
            double error = fabs(T_new[i][j] - old_T);
            if (error > *max_diff) {
                *max_diff = error;
            }

            // Update T[i][j] immediately for Gauss-Seidel
            T[i][j] = T_new[i][j];
        }
    }
}

// Function to write grid to file
void writeGridToFile(FILE *file) {
    for (int i = 0; i < MX; i++) {
        for (int j = 0; j < NY; j++) {
            fprintf(file, "%f ", T[i][j]);
        }
        fprintf(file, "\n");
    }
    fprintf(file, "\n\n"); // Two blank lines between datasets for gnuplot
}

int main() {
    double max_diff;
    int iter = 0;

    printf("Number of cells in the X-axis: %d\n", MX);
    printf("Number of cells in the Y-axis: %d\n", NY);

    // Initialize temperature array
    for (int i = 0; i < MX; i++) {
        for (int j = 0; j < NY; j++) {
            T[i][j] = 0.0; // Initialize all temperatures to 0
        }
    }

    // Set boundary conditions
    setBoundaryConditions();

    // Open output file
    FILE *outputFile = fopen("temperature_field.txt", "w");
    if (outputFile == NULL) {
        printf("Error opening output file!\n");
        return 1;
    }

    // Gauss-Seidel iteration
    do {
        gaussSeidelIteration(&max_diff);

        iter++;

        // Write grid to file every iteration
        writeGridToFile(outputFile);

    } while (max_diff > ERROR && iter < Max_iteration);

    // Write final state
    writeGridToFile(outputFile);

    // Close output file
    fclose(outputFile);

    if (iter >= Max_iteration) {
        printf("Failed to converge after %d iterations\n", Max_iteration);
    } else {
        printf("Converged after %d iterations\n", iter);
    }

    return 0;
}
