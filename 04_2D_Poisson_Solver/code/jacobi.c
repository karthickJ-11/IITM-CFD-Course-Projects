// 2-D Poisson equation, unit square, jacobi iteration.
// Original author: Karthick J (OE23S018), OE5450 Assignment 3 (code taken from the submitted report).
// Corrections are marked [FIX]; additions [ADDED].
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
#define CASE 1           // [ADDED] boundary-condition case: 1 or 2 (see problem statement)
#define DTDX_RIGHT -1.0  // [ADDED] Case 2: prescribed dT/dx on x = 1
#define WRITE_EVERY 100  // [FIX] write the field every WRITE_EVERY iterations (was every iteration:
                         //  up to ~0.7 GB per run for delta = 0.01)

float T[MX][NY], T_new[MX][NY];

// [FIX: orientation. The problem statement prescribes (Case 1) T = 10 on x = 0 and x = 1 and
//  T = 0 on y = 0 and y = 1; the original code applied T_max on y = 0, 1 and T_min on x = 0, 1,
//  i.e. the problem rotated by 90 degrees. (The Jacobi listing also wrote T[MX-1][NY], one
//  element past the end of the array, so its right wall was never set.)]
// [ADDED: Case 2 -- T = 0 on y = 0, y = 1 and x = 0; dT/dx = -1 on x = 1 -- which was used for the
//  Case 2 results in the original table but is not in the submitted listing.]
// Neumann condition, second-order one-sided: (3 T_N - 4 T_{N-1} + T_{N-2}) / (2 dx) = dT/dx
void applyNeumann() {
    for (int j = 1; j < NY - 1; j++) {
        T[MX - 1][j] = (4.0 * T[MX - 2][j] - T[MX - 3][j] + 2.0 * delta_x * DTDX_RIGHT) / 3.0;
    }
}

void setBoundaryConditions() {
    int i, j;
    // Bottom (y = 0) and top (y = 1) boundaries: T = T_min in both cases
    for (i = 0; i < MX; i++) {
        T[i][0] = T_min;
        T[i][NY - 1] = T_min;
    }
    // Left (x = 0) and right (x = 1) boundaries
    for (j = 0; j < NY; j++) {
        T[0][j] = (CASE == 1) ? T_max : T_min;
        if (CASE == 1) T[MX - 1][j] = T_max;
    }
    if (CASE == 2) applyNeumann();
}

// Function to perform Jacobi iteration poisson equation 
void jacobiIteration() {
    int i, j;
    float q1 = 1.0, q2 = -1.0;
    // [FIX: (int)(0.45/0.01) = 44 because 0.45/0.01 = 44.999...; round to the nearest node]
    int sx1 = (int)(0.45 / delta_x + 0.5), sy1 = (int)(0.5 / delta_y + 0.5);
    int sx2 = (int)(0.55 / delta_x + 0.5), sy2 = (int)(0.5 / delta_y + 0.5);

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

        if (CASE == 2) applyNeumann(); // [ADDED] update the Neumann boundary after each sweep

        iter++;

        // Write grid to file every WRITE_EVERY iterations  [FIX]
        if (iter % WRITE_EVERY == 0) {
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
