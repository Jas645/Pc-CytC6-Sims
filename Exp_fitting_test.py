import numpy as np
import matplotlib.pyplot as plt
import os
import Surface_confined_inference as sci

# Load the data
filename = "-300_to_500mV_29.80mVs-1_15.98Hz_200mVamp_pH7.5_5C_exp_"
data = np.loadtxt(r'/users/jas645/experimental/Pc_pH7.5/' + filename)

# Extract columns
time = data[::16, 0]
current = data[::16, 1]
potential = data[::16, 2]

# Call the inference function

"""estimated_parameters, fitted_parameters, estimated_simulation, fitted_simulation = sci.infer.get_input_parameters(
    time, 
    potential, 
    current, 
    "FTACV",
    optimise=True,
    return_sim_values=True,
    sinusoidal_phase=False,
    sigma=0.075,
    runs=20
)"""


fitted_parameters={'E_start': np.float64(-0.30039879816754445), 'E_reverse': np.float64(0.5004849796959054), 'omega': np.float64(15.925615672339024), 'phase': np.float64(0.012104978977064047), 'delta_E': np.float64(0.1987892216532862), 'v': np.float64(0.029836144139503893)}

print (fitted_parameters)
input_parameters = fitted_parameters
input_parameters["Temp"] = 278
input_parameters["N_elec"] = 1
input_parameters["area"] = 3e-2
input_parameters["Surface_coverage"] = 1e-11

slurm_class = sci.SingleSlurmSetup(
	"FTACV",
	input_parameters)
slurm_class.dispersion_bins=[20]
slurm_class.GH_quadrature=True
slurm_class.optim_list = ["E0_mean","E0_std", "k0", "Ru", "gamma", "Cdl", "CdlE1","CdlE2", "CdlE3", "alpha","phase"]	
slurm_class.boundaries = {"k0": [5, 5000], 
                            "E0_mean":[0, 0.3],
                            "E0_std":[1e-3, 0.15],
                            "Cdl": [1e-6, 5e-4],
                            "gamma": [1e-12, 8e-10],
                            "Ru": [0.1, 2000],
                            "alpha":[0.4, 0.6],
                            "CdlE1":[-1e-2, 1e-2],
                            "CdlE2":[-5e-3, 4e-3],
                            "CdlE3":[-5e-5, 5e-5],
                            "omega":[0.8*fitted_parameters["omega"], 1.2*fitted_parameters["omega"]],
                            "phase":[0, 2*np.pi]
                            }


#simcurrent = slurm_class.dim_i(slurm_class.Dimensionalsimulate([0.145, 0.04, 500, 1000, 1e-11, 1e-5, 0.5], time))

"""fig, axis = plt.subplots()
axis.plot(time, current + random_noise)
axis.plot(time, current)
plt.xlabel("Time (s)")
plt.ylabel("Current (A)")
plt.show()"""
"""
plot_dict = dict(Data_data={"time":time, "current":current, "potential":potential, "harmonics":list(range(1,9))},
	plot_func = np.abs, hanning = True, filter_val = 0.5, xlabel = "Time (s)", ylabel = "Current (A)", remove_xaxis = True)
plot_dict["Sim_data"] = {"time":time, "current":simcurrent, "potential":potential, "harmonics":list(range(1,9))}
"plt.plot(time, current)"
sci.plot.plot_harmonics(**plot_dict)
plt.show()
"""

slurm_class.Fourier_fitting=True
slurm_class.Fourier_window="hanning"
slurm_class.top_hat_width=0.5
slurm_class.Fourier_function="abs"
slurm_class.Fourier_harmonics=list(range(4, 8))



slurm_class.setup(
    datafile=r'/users/jas645/experimental/Pc_pH7.5/'+filename,
    cpu_ram="12G",
    time="0-48:00:00",
    runs=10, 
    threshold=1e-8, 
    unchanged_iterations=200,
    results_directory="Data_inference_1_7.5_15.98Hz_plastocyanin_Har4-7",
    save_csv=False,
    debug=False,
    run=True
)


"""# Plotting
fig, axis = plt.subplots()
twinx = axis.twinx()
axis.plot(time, potential, label="Data", color = "pink")
axis.plot(time, fitted_simulation, label = "Best Fit", color = "purple")
twinx.plot(time, potential-fitted_simulation, label="Residual", color="cyan")
plt.legend()
plt.title("Potential Check of" + filename)
folder_path = r'/users/jas645/code/Pc-CytC6-Sims/Figures'
figurename = "Pot_Check" + filename + ".png"
os.makedirs(folder_path, exist_ok=True)

plt.show()
#plt.savefig(os.path.join(folder_path, figurename))


# Print fitted parameters
print("experiments_dict['{0}']['{1}']=".format(CONDITION_1, CONDITION_2), fitted_parameters)"""
