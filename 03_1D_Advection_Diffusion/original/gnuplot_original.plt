# plot_all_dye.gnu
set terminal pngcairo size 800,600 enhanced font 'Arial,12'
set output 'dye_concentration_1_50_s.png'
set title "Dye Concentration Evolution (Advection-Diffusion)\nCFL=0.05, Diffusion Number=0.125, velocity = 1.50m/s"
set xlabel "Position (m)"
set ylabel "Concentration (g/cm³)"
set grid

plot for [i=0:9] 'results.txt' index i using 1:2 with lines title sprintf("Time = %d s", (i+1)*1)

