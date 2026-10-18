clear; clc;

% Define constants
L = 100;     % Length of the domain (cm)
CX = 51;   % Number of grid points
alpha = 1; % Thermal diffusivity (m²/s)
dt = 0.001;% Time step (s)
T0 = 1.0;  % Temperature at x=0 (boundary condition)
TL = 0.0;  % Temperature at x=L (boundary condition)
t = 10.0;  % Total simulation time (s)

% Calculate grid size
dx = L / (CX - 1);
fprintf('dx: %f\t', dx);
fprintf('dt: %f\t', dt);

% Calculate number of time steps
Nt = floor(t / dt);
fprintf('Nt: %d\t', Nt);

% Calculate gamma
gamma = alpha * (dt / (dx^2));
fprintf('\ngamma: %f\t', gamma);

% Initialize temperature grid
grid = zeros(1, CX);
DT_values = []; % Array to store temperature changes

% Open file for writing
fileID = fopen('temperature_output.txt', 'w');
fprintf(fileID, 'DX\tDT\n'); % Write header to the file

% Time loop
for j = 0:Nt
    Time = j * dt;
    fprintf('\nTime: %.2f Seconds\n', Time);
    
    % Impose boundary conditions
    grid(1) = T0;       % Left side boundary condition
    grid(end) = TL;     % Right side boundary condition
    
    % Update interior points
    for i = 2:CX-1
        grid(i) = (gamma * grid(i-1)) + ((1 - (2 * gamma)) * grid(i)) + (gamma * grid(i+1));
        fprintf('\nGrid(%d) = %.2f\n', i, grid(i));
    end
    
    % Calculate DT for interior points and store them
    for i = 2:CX-1
        DT_values(i-1, j+1) = grid(i); % Store temperature values for plotting later
    end
    
    % Write grid points and temperatures to the file
    fprintf(fileID, 'Time: %.2f seconds\n', Time);
    for i = 2:CX-1
        fprintf(fileID, '%.4f\t%.4f\n', (i-1)*dx, grid(i));
    end
    fprintf(fileID, '\n'); % New line for separation between time steps
end

fclose(fileID); % Close the file

% Prepare data for plotting DT vs DX at final time step
final_DT = DT_values(:, end);  % Get temperature values at last time step

% Plot DX vs DT using final temperature values
figure;
plot((1:CX-2)'*dx, final_DT, '-o'); 
xlabel('DX (m)');
ylabel('DT (°C)');
title('DX vs DT (dt=0.001, t=10.0s)');
grid on;