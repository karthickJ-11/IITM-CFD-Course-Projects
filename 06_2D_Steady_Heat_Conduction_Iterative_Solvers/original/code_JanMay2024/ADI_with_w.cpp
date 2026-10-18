//ADI WITH RELAXATION FACTOR

#include<iostream>
#include<math.h>
#include<fstream>
#define NX 21
#define NY 41

double LX = 1;
double LY = 2;
double T_top_wall = 100;
double T_wall = 0;
double T[NX][NY];
double delta_x, delta_y, beta, omega;

void Boundary_condition()
{
    for (int i=0; i<NX; i++)
    {
        T[i][0]= T_wall;
    }
    for (int i=0; i < NX; i++)
    {
        for (int j=0; j < (NY-1); j++)
        {
            T[i][j]=T_wall;
        }
    }
    for (int i=0; i < (NX-1); i++)
    {
        for (int j=0; j < NY; j++)
        {
            T[i][j]=T_wall;
        }
    }
    for (int i=0; i < NX; i++)
    {
        T[i][NY-1]= T_top_wall;
    }
}

int main()
{
    clock_t start, end;
    double CPU_time_used;
    start = clock();
    delta_x = (LX/ NX);
    delta_y = (LY/ NY);
    beta = (delta_x / delta_y);
    omega = 1.5; // Set the relaxation factor here

    std::cout << "Cell size in X axis: " << delta_x << std::endl;
    std::cout << "Cell size in Y axis: " << delta_y << std::endl;
    std::cout << "Beta: " << beta << std::endl;
    std::cout << "Omega: " << omega << std::endl;

    // Initialize the grid size
    for(int i=0; i< NX; i++)
    {
        for(int j=0; j<NY; j++)
        {
            T[i][j]=0;
        }
    }

    // First iteration begin
    for(int j=1; j< NY-1; j++)
    {
        T[0][j]=( ((1-omega)*(T[0][j])) + (omega/(1+beta*beta))*(T[1][j]));
    }

    for(int i=1; i< NX-1; i++)
    {
        T[i][1]=( ((1-omega)*(T[i][1])) + (omega/(1+beta*beta))*(T[i-1][1] + T[i+1][1]));
        for(int j=2; j< NY-1; j++)
        {
            T[i][j]=( ((1-omega)*(T[i][j])) + (omega/(1+beta*beta))*(T[i-1][j] + T[i+1][j] + beta*beta*(T[i][j-1] + T[i][j+1])));
        }
        T[i][NY-2]=( ((1-omega)*(T[i][NY-2])) + (omega/(1+beta*beta))*(T[i-1][NY-2] + T[i+1][NY-2] + beta*beta*(T[i][NY-3])));
    }

    for(int j=1; j< NY-1; j++)
    {
        T[NX-1][j]=( ((1-omega)*(T[NX-1][j])) + (omega/(1+beta*beta))*(T[NX-2][j]));
    }

    Boundary_condition();

    // Iteration start
    double tolerance = 0.01;
    double iteration = 0.5;
    double T_new[NX][NY];
    
    std::ofstream ofile("final_temp.txt");

    while(true)
    {
        iteration = iteration+0.5; // Update iteration level

        // Perform ADI calculations for X direction
        for(int i=1; i< NX-1; i++)
        {
            for(int j=1; j< NY-1; j++)
            {
                T_new[i][j]=( ((1-omega)*(T[i][j])) + (omega/(1+beta*beta))*(T[i-1][j] + T[i+1][j]));
            }
        }

        // Perform ADI calculations for Y direction
        for(int j=1; j< NY-1; j++)
        {
            for(int i=1; i< NX-1; i++)
            {
                T[i][j] = T_new[i][j] + (omega/(1+beta*beta))*(T_new[i][j-1] + T_new[i][j+1] + beta*beta*(T_new[i-1][j] + T_new[i+1][j]));
            }
        }

        // Check if the desired tolerance level has been reached
        int tolerance_not_reached_count = 0;
        for(int i=0; i< NX; i++)
        {
            for(int j=0; j<NY; j++)
            {
                double delta_T = fabs(T_new[i][j] - T[i][j]);
                if(delta_T > tolerance)
                {
                    tolerance_not_reached_count++;
                    break;
                }
            }
            if(tolerance_not_reached_count > 0)
            {
                break;
            }
        }

        // If the tolerance level has been reached, break out of the loop
        if(tolerance_not_reached_count == 0)
        {
            break;
        }

        // Update the temperature grid with the new values
        for(int i=0; i< NX; i++)
        {
            for(int j=0; j<NY; j++)
            {
                T[i][j] = T_new[i][j];
                ofile << i << "\t" << j << "\t" << T[i][j] << std::endl;
            }
        }
    }

    std::cout << "Tolerance reached after " << iteration << " iterations." << std::endl;
    ofile.close();
    end = clock();
    CPU_time_used = static_cast<double>(end - start) / CLOCKS_PER_SEC;
    // Calculate the elapsed time in seconds

    std::cout << "CPU_time_used: " << CPU_time_used << " seconds" << std::endl;
    return 0;
}

