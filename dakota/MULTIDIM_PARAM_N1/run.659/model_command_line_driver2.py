import sys
import os
import numpy as np
from subprocess import call
from yaml import safe_load
from bug_model.simulate_bug_decay import run_exp_three_times_and_bin
from bug_model.simulate_bug_decay import bin_output_data, DATA_DIR

# This script much indebted to Katy Barnhart's exceptional tutorial on
# Dakota for calibration; see
# https://github.com/kbarnhart/calibration_with_dakota_clinic

# STEP 1: Use Dakota-created input files to prepare for run
input_template = "input_template.yml"
inputs = "inputs.yml"
call(["dprepro", sys.argv[1], input_template, inputs])
call(['rm', input_template])

# STEP 2: Run model
# Load parameters from the yaml formatted input.
with open(inputs, "r") as f:
    params = safe_load(f)
    R = params["R"]
    S = params["S"]
    F = params["F"]

bin_intervals = np.loadtxt(
    os.path.join(DATA_DIR, 'real_data_bin_intervals.txt')
)

# launch the simulation
binned_sim_bfr = run_exp_three_times_and_bin(
    theta=(R, S, F), x=bin_intervals, n='from_data', repeats=3, stochastic=True
)

# Step 3: Write output in format Dakota expects
# Each of the metrics listed in the Dakota .in file needs to be written to
# the specified output file given by sys.argv[2]. This is how information is
# sent back to Dakota.

# calc the rmse
real_data = np.loadtxt(
    os.path.join(DATA_DIR, 'real_data_bin_means.txt')
)
# repo lengths are used directly by model and remain in order so simply now
rmse = (np.mean((real_data - binned_sim_bfr) ** 2)) ** 0.5

# Write it to the expected file.
with open(sys.argv[2], "w") as fp:
    fp.write(str(rmse))
