# CFD and Numerical Methods Course Projects – IIT Madras

**Karthick J** · M.S. (by Research), Department of Ocean Engineering, IIT Madras

Seven computational fluid dynamics and numerical-methods projects written from scratch in C and CUDA C, each with corrected code, results, a technical report (PDF) and a video of the simulation. Courses: AM5630 *Foundations of Computational Fluid Dynamics* and OE5450 *Numerical Techniques in Ocean Hydrodynamics*.

## Projects

| # | Project | Notes | Report | Video |
|---|---|---|---|---|
| 01 | GPU FVM non-Newtonian blood flow (CUDA) | [README](01_GPU_FVM_Arterial_Blood_Flow/README.md) | [PDF](01_GPU_FVM_Arterial_Blood_Flow/report/report.pdf) | [blood_flow_LAD_startup.mp4](01_GPU_FVM_Arterial_Blood_Flow/video/blood_flow_LAD_startup.mp4) |
| 02 | GPU lid-driven cavity, vorticity–stream function (CUDA) | [README](02_GPU_Lid_Driven_Cavity_CUDA/README.md) | [PDF](02_GPU_Lid_Driven_Cavity_CUDA/report/report.pdf) | [cuda_cavity_vorticity_streamfunction.mp4](02_GPU_Lid_Driven_Cavity_CUDA/video/cuda_cavity_vorticity_streamfunction.mp4) |
| 03 | 1-D advection–diffusion | [README](03_1D_Advection_Diffusion/README.md) | [PDF](03_1D_Advection_Diffusion/report/report.pdf) | [advection_diffusion_u0_1_1.5_2.mp4](03_1D_Advection_Diffusion/video/advection_diffusion_u0_1_1.5_2.mp4) |
| 04 | 2-D Poisson solver: Jacobi, Gauss–Seidel, SUR | [README](04_2D_Poisson_Solver/README.md) | [PDF](04_2D_Poisson_Solver/report/report.pdf) | [poisson_iterations_jacobi_gs_sur.mp4](04_2D_Poisson_Solver/video/poisson_iterations_jacobi_gs_sur.mp4) |
| 05 | Lid-driven cavity, SMAC method | [README](05_Lid_Driven_Cavity_SMAC/README.md) | [PDF](05_Lid_Driven_Cavity_SMAC/report/report.pdf) | [smac_cavity_startup_to_steady.mp4](05_Lid_Driven_Cavity_SMAC/video/smac_cavity_startup_to_steady.mp4) |
| 06 | 2-D steady heat conduction, six iterative solvers | [README](06_2D_Steady_Heat_Conduction_Iterative_Solvers/README.md) | [PDF](06_2D_Steady_Heat_Conduction_Iterative_Solvers/report/report.pdf) | [iterative_solvers_convergence.mp4](06_2D_Steady_Heat_Conduction_Iterative_Solvers/video/iterative_solvers_convergence.mp4) |
| 07 | 1-D transient heat conduction, explicit FTCS | [README](07_1D_Transient_Heat_Conduction_Explicit_FDM/README.md) | [PDF](07_1D_Transient_Heat_Conduction_Explicit_FDM/report/report.pdf) | [stable](07_1D_Transient_Heat_Conduction_Explicit_FDM/video/conduction_FTCS_dt0.1_every_step.mp4) · [unstable](07_1D_Transient_Heat_Conduction_Explicit_FDM/video/conduction_FTCS_dt2.5_instability.mp4) |

## Folder layout

Each project folder is self-contained:

| folder | contents |
|---|---|
| `code/` | corrected source (changes marked `[FIX]`, `[ADDED]`, `[CHANGED]`) and the Python post-processing |
| `original/` | the code, report and plots exactly as submitted (unchanged) |
| `results/` | figures, `summary.txt`, and raw run outputs (zipped as `raw_runs.zip` where large) |
| `report/` | technical LaTeX report (`report.tex`, `report.pdf`) with flowchart, code review, validation |
| `run.sh` | rebuilds and reruns everything for that project, then regenerates the figures |
| `video/` | time-step / iteration animations (MP4) of the simulation, made by `make_video.sh` + `code/make_video.py` |

| # | project | course | language | key result |
|---|---|---|---|---|
| 01 | GPU FVM non-Newtonian (Bird-Carreau) blood flow in the LAD artery | OE5450, Jan-May 2025 | CUDA C | developed profile within 0.2 % of exact Carreau solution; Newtonian check to 2.6e-6 |
| 02 | GPU vorticity-stream function lid-driven cavity, Re = 100 | OE5450, Jan-May 2025 | CUDA C | psi_min = -0.10350; 3-grid study, observed order 2.01-2.02 |
| 03 | 1-D advection-diffusion of a dye release | OE5450, Jan-May 2025 | C | numerical diffusion of upwind FTBS quantified (D_eff vs theory) |
| 04 | 2-D Poisson equation: Jacobi, Gauss-Seidel, SUR | OE5450 (NTH), Jan-May 2025 | C | iteration counts vs theory, termination error |
| 05 | Lid-driven cavity with the SMAC projection method, Re = 100 | AM5630, Jan-May 2024 | C | psi_min = -0.10337 vs Ghia -0.10342 |
| 06 | 2-D steady heat conduction: PGS, PSOR, LGS, LSOR, ADI | AM5630, Jul-Nov 2024 | C | omega_opt study, exact heat balance |
| 07 | 1-D transient heat conduction, explicit FTCS | AM5630, Jul-Nov 2024 | C | second-order grid convergence, stability limit gamma <= 1/2 |

## Rerunning
    bash run_all.sh            # every project, CPU (~45 min); CUDA projects also on the GPU if nvcc + an NVIDIA GPU exist
    bash 01_GPU_FVM_Arterial_Blood_Flow/run.sh gpu    # GPU runs only for one project

Videos: `bash NN_name/make_video.sh` (needs ffmpeg) reruns the code with frequent output and rebuilds `video/*.mp4`.

Requirements: gcc/g++, python3 with numpy, scipy, matplotlib; pdflatex for the reports; nvcc (CUDA toolkit) for the GPU runs.
The CUDA projects can also run without a GPU: `tools/emulate.sh` executes the unmodified kernels serially on the CPU
(this is how the results in `results/` were produced). On a GPU machine `run.sh` writes the GPU wall times to `results/timing.txt`.
