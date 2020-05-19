import sys
import numpy as np
from subprocess import call
from simulate_bug_decay import run_with_exponential_num_bugs_floats_in

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

# launch the simulation
nums_caught, bug_rates = run_with_exponential_num_bugs_floats_in(
    R, S, F, 'from_data'
)

# Step 3: Write output in format Dakota expects
# Each of the metrics listed in the Dakota .in file needs to be written to
# the specified output file given by sys.argv[2]. This is how information is
# sent back to Dakota.

# calc the rmse
real_bug_rates = np.loadtxt('../all_real_data_bug_find_rate')
# repo lengths are used directly by model and remain in order so simply now
rmse = (np.mean((real_bug_rates - bug_rates) ** 2)) ** 0.5

# Write it to the expected file.
with open(sys.argv[2], "w") as fp:
    fp.write(str(rmse))
