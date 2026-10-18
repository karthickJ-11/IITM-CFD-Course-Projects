% Constants
M = 41;  % Number of cells in Y axis (height)
N = 21;  % Number of cells in X axis (width)
T_BOTTOM = 100.0; % Bottom boundary temperature
T_OTHER = 30.0;   % Other boundaries temperature
ERROR_THRESHOLD = 0.01; % Convergence criteria
K = 50.0; % Thermal conductivity (W/mK)

% Initialize temperature grid
T = zeros(M, N);

% Set boundary conditions
function T = setBoundaryConditions(T, M, N, T_BOTTOM, T_OTHER)
    T(M, :) = T_BOTTOM; % Bottom wall
    T(1, :) = T_OTHER;  % Top wall
    T(:, 1) = T_OTHER;  % Left wall
    T(:, N) = T_OTHER;  % Right wall
end

% Calculate heat transfer along the bottom boundary
function total_heat_transfer = calculateHeatTransfer(T, M, N, K)
    total_heat_transfer = 0.0;
    dx = 1.0 / (N - 1); % Assuming unit length in X direction
    for j = 1:(N - 1)
        total_heat_transfer = total_heat_transfer + (T(M, j) + T(M, j + 1)) / 2 * dx;
    end
    total_heat_transfer = total_heat_transfer * K; % Multiply by thermal conductivity
end

% Main program
T = setBoundaryConditions(T, M, N, T_BOTTOM, T_OTHER);
max_error = inf;
iteration_count = 0;

% Prepare figure for plotting and video recording
figure('Position', [100, 100, 840, 630]); % Set figure size to match video frame size
videoWriter = VideoWriter('temperature_distribution.mp4', 'MPEG-4'); % Create a video file
videoWriter.FrameRate = 2; % Set frame rate (frames per second)
open(videoWriter); % Open the video writer

omega = 1.5; % Relaxation factor

while max_error > ERROR_THRESHOLD
    max_error = 0.0;
    iteration_count = iteration_count + 1;

    % Sweep along the x-axis 
    for i = 2:(M - 1)
        for j = 2:(N - 1)
            old_temp = T(i, j);
            T(i, j) = (1 - omega) * old_temp + omega * (0.25 * (T(i + 1, j) + T(i - 1, j) + T(i, j + 1) + T(i, j - 1)));
            error = abs(T(i, j) - old_temp);
            if error > max_error
                max_error = error;
            end
        end
    end

    % Sweep along the y-axis 
    for j=2:(N - 1)
        for i=2:(M - 1)
            old_temp = T(i, j);
            T(i, j) = (1 - omega) * old_temp + omega * (0.25 * (T(i + 1, j) + T(i - 1, j) + T(i, j + 1) + T(i, j - 1)));
            error = abs(T(i, j) - old_temp);
            if error > max_error
                max_error = error;
            end
        end
    end

    % Plot current temperature grid as a heatmap
    imagesc(T); 
    colorbar;
    title(['Temperature Distribution After Iteration ' num2str(iteration_count)]);
    xlabel('X Axis');
    ylabel('Y Axis');
    caxis([T_OTHER T_BOTTOM]); % Set color axis limits based on boundary conditions
    
    % Display updated temperatures on the grid
    hold on;
    for i = 1:M
        for j = 1:N
            text(j, i, sprintf('%.1f', T(i,j)), 'Color', 'k', 'FontSize', 8, 'HorizontalAlignment', 'Center');
        end
    end
    
    hold off;

    pause(0.5); % Pause to visualize changes

    % Capture the current frame and write it to the video file
    frame = getframe(gcf); 
    writeVideo(videoWriter, frame);
end

% Final plot after convergence
imagesc(T);
colorbar;
title('Final Temperature Distribution');
xlabel('X Axis');
ylabel('Y Axis');
caxis([T_OTHER T_BOTTOM]); 

% Display updated temperatures on the final grid
hold on;
for i = 1:M
    for j = 1:N
        text(j, i, sprintf('%.1f', T(i,j)), 'Color', 'k', 'FontSize', 8, 'HorizontalAlignment', 'Center');
    end
end

hold off;

% Capture the final frame and write it to the video file
frame = getframe(gcf); 
writeVideo(videoWriter, frame);

% Close the video file after writing all frames
close(videoWriter);

% Calculate and display heat transfer along the bottom boundary
heatTransfer = calculateHeatTransfer(T, M, N, K);
fprintf('Total Heat Transfer along Bottom Boundary: %.2f W\n', heatTransfer);