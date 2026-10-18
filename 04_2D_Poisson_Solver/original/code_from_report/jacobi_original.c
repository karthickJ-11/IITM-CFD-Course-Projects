//*******************JACOBI ITERATION METHOD*********************//
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define L 1.0 //x and y axis length of the domain 
#define delta_x 0.01 //length of the cell size
#define delta_y 0.01 //length of the cell size
#define MX ((int)(L/delta_x)+1) // number of cells in x axis 
#define NY ((int)(L/delta_y)+1) // number of cells in y axis
#define ERROR 0.0001 //error for the iteration
#define T_max 10.0 //maximum temperature of the boundary condition in degrees
#define T_min 0.0 //Minimum temperature of the boundary condition in degrees
#define Max_iteration 10000 //maximum iteration for the convergence

float T[MX][NY], T_new[MX][NY];

// Function to set boundary conditions
void setBoundaryConditions() {
    int i, j;
    // Top side Boundary condition 
    for (i = 0; i < MX; i++) {
        T[i][NY-1] = T_max;
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
        T[MX-1][NY]= T_min;
    }
}

// Function to perform Jacobi iteration poisson equation 
void jacobiIteration() {
    int i, j;
    float q1 = 1.0, q2 = -1.0;
    int sx1 = (int)(0.45 / delta_x), sy1 = (int)(0.5 / delta_y);
    int sx2 = (int)(0.55 / delta_x), sy2 = (int)(0.5 / delta_y);

    for (i = 1; i < MX-1; i++) {
        for (j = 1; j < NY-1; j++) {
            float source = 0;
            if (i == sx1 && j == sy1) source = q1;
            if (i == sx2 && j == sy2) source = q2;
            T_new[i][j] = 0.25 * (T[i+1][j] + T[i-1][j] + T[i][j+1] + T[i][j-1] - (delta_x * delta_x) * source);
        }
    }
}

// Function to write grid to file in gnuplot format
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
    float max_diff;
    int iter = 0;
    
    printf("Number of cells in the X-axis: %d\n", MX);
    printf("Number of cells in the Y-axis: %d\n", NY);
    
    // Initialize temperature array
    for (int i = 0; i < MX; i++) {
        for (int j = 0; j < NY; j++) {
            T[i][j] = 0.0;
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

    // Jacobi iteration
    do {
        jacobiIteration();
        
        // Calculate error and convergence criteria
        max_diff = ERROR;
        for (int i = 1; i < MX-1; i++) {
            for (int j = 1; j < NY-1; j++) {
                float error = fabs(T_new[i][j] - T[i][j]);
                if (error > max_diff) {
                    max_diff = error;
                }
                T[i][j] = T_new[i][j];
            }
        }

        iter++;

        // Write grid to file every 1 iterations
        if (iter % 1 == 0) {
            writeGridToFile(outputFile);
        }
        
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
