import sys
import os
import numpy as np
from subprocess import call
from yaml import safe_load
from bug_model.simulate_bug_decay import run_with_exponential_num_bugs_floats_in
from bug_model.simulate_bug_decay import DATA_DIR
from bug_model.driver import create_bins

BINNING_SCALE = 20
LOW_COMMITS = 10
REPEATS = 3

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

# load the data
real_bfr = np.loadtxt(os.path.join(DATA_DIR,
                                   'all_real_data_bug_find_rate.txt'))
real_commits = np.loadtxt(os.path.join(DATA_DIR,
                                       'all_real_data_total_commits.txt'))
real_bins, real_bin_counts, real_binned_bfr = create_bins(
    BINNING_SCALE, real_commits, real_bfr
)


# launch the simulation
avg_sim_bin_bfr = np.zeros_like(real_binned_bfr, dtype=float)
avg_sim_std_low_commits = 0
for i in range(REPEATS):
    num_bugs, bug_rate, num_commits = run_with_exponential_num_bugs_floats_in(
        R, S, F, num_realisations='from_data', stochastic=True
    )
    bins, bin_counts, sim_bin_bfr = create_bins(
        BINNING_SCALE, num_commits, bug_rate
    )
    avg_sim_bin_bfr += sim_bin_bfr / float(REPEATS)
    low_commit_repos = num_commits < LOW_COMMITS
    avg_sim_std_low_commits += np.std(
        bug_rate[low_commit_repos]
    ) / float(REPEATS)


# Step 3: Write output in format Dakota expects
# Each of the metrics listed in the Dakota .in file needs to be written to
# the specified output file given by sys.argv[2]. This is how information is
# sent back to Dakota.

# calc the rmse
# repo lengths are used directly by model and remain in order so simply now
rmse = (np.mean((real_binned_bfr - avg_sim_bin_bfr) ** 2)) ** 0.5

# calc the diff_std_at_low_commits
# note we have to use each individual sim, not the std of the averaged run
real_std_low_commits = np.std(real_bfr[real_commits < LOW_COMMITS])
diff_std_at_low_commits = np.abs(real_std_low_commits - avg_sim_std_low_commits)

# Write both to the expected file.
with open(sys.argv[2], "w") as fp:
    fp.write(str(rmse) + '\n' + str(diff_std_at_low_commits))
