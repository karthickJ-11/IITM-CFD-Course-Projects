// 1D heat convection Equation

#include <stdio.h>
#include <math.h>
#define CX 51// Number of grid points

int main() {
    // Declare the variables
    double L = 100.0; // cm   [FIX 1: was int -> integer division risk in dx = L/(CX-1)]
    double grid[CX];
    double alpha = 1;        // m²/s
    double dt = 0.001;         // sec
    double T0 = 1.0;         // Boundary condition at x=0 length
    double TL = 0.0;         // Boundary condition at x=L length
    double TN[CX];           // Temperature at new grid points
    double t = 10.0;          // Total time for simulation
    int DX,DT; // for output file notation
    // Cell size calculation
    double dx;               // Size of the cell in x-axis
    dx = L / (CX - 1);
    printf("dx: %lf\t", dx);
    printf("dt: %lf\t", dt);

    // Time steps calculations

    int Nt = (int)(t / dt);  // Number of time steps, cast to int for loop control
    printf("Nt: %d\t", Nt);

    // Gamma Calculation
    double gamma = alpha * (dt / (dx * dx));
    printf("\ngamma: %f\t", gamma);

    // Initial condition: Set all temperatures to zero initially.
    for (int i = 0; i < CX; i++) {
        grid[i] = 0.0;
    }

    printf("\nInitial Temperature at grid Points:\n");
    for (int i = 0; i < CX; i++) {
        printf("grid[%d] = %.2f\n", i, grid[i]);
    }

	// Open a file for writing
    FILE *outputFile;
    outputFile = fopen("temperature_output.txt", "w");

    if (outputFile == NULL) {
        printf("Error opening file!\n");
        return 1;
    }

    // Write header to the file
	fprintf(outputFile, "DX\tDT\n"); // Header row


	// [FIX 2: write the initial condition at t = 0 before any update, and label each
	//  stored profile with the time it actually represents (after j updates)]
	grid[0] = T0; grid[CX - 1] = TL;
	fprintf(outputFile, "Time: %.4f seconds\n", 0.0);
	for (int i = 0; i < CX; i++) fprintf(outputFile, "%d\t%.6f\n", i, grid[i]);
	fprintf(outputFile, "\n");

	// Time loop initializing
	for (int j = 1; j <= Nt; j++) {
		double Time = j * dt; // Calculating the run time (state after j steps)
		printf("\nTime: %.2f Seconds\n", Time);

		// Impose boundary conditions
		grid[0] = T0;       // Left side Boundary condition
		grid[CX - 1] = TL;  // Right side Boundary Condition

		for (int i = 1; i < CX - 1; i++) {
			TN[i] = (gamma * grid[i - 1]) + ((1 - (2 * gamma)) * grid[i]) + (gamma * grid[i + 1]);
			printf("\nTN[%d] = %.2f\n", i, TN[i]);
		}

		for (int i = 1; i < CX - 1; i++) {
			grid[i] = TN[i]; // Update temperature values for next iteration
		}

        // Write grid points and temperatures to the file after updating temperatures
        fprintf(outputFile, "Time: %.4f seconds\n", Time);
        for (int i = 0; i < CX; i++) {
      //    fprintf(outputFile, "Grid%d: Temperature %.2f\n", i, grid[i]);
             fprintf(outputFile, "%d\t%.6f\n", i, grid[i]);  // [FIX 3: 2 -> 6 decimals; 0.01 resolution hid the early-time profile]
        }
        fprintf(outputFile, "\n"); // New line for separation between time steps
	}

	fclose(outputFile); // Close the file after writing

	return 0;
}


