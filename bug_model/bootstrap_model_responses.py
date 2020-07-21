# this follows on from dakota_plotters, and visualises the range of poss
# model responses under our uncertainty brackets.

import os
import pandas as pd
from matplotlib.pyplot import plot, figure, show
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde, expon, geom
import numpy as np
from bug_model.simulate_bug_decay import run_with_exponential_num_bugs_floats_in
from bug_model.simulate_bug_decay import DATA_DIR
from bug_model.driver import create_bins

# copied from dakota/template_dir/model_command_line_driver4
# should set up a general contstant import file
BINNING_SCALE = 20
BINNING_SCALE_STD = 50
REPEATS = 3

# redo the actual bootstrapping, following dakota_plotters.py:
DATFILE = '../dakota/dakota_gridtraverse_N1.dat'
INPARAMS = ["R", "S", "F"]
# OUTPARAMS = ["rmse","diff_std_at_low_commits"]
OUTPARAMS = ["rmse","rmse_std"]

df = pd.read_csv(DATFILE, engine='python', delim_whitespace=True)
# gaussian_kde fit to locate most likely solutions:
R = df[INPARAMS[0]]
S = df[INPARAMS[1]]
F = df[INPARAMS[2]]
RMSE = df[OUTPARAMS[0]]
RMSE_TOT = df[OUTPARAMS[0]] + df[OUTPARAMS[1]]

values = np.vstack([R, S, F])
kernel = gaussian_kde(values, weights=1./RMSE)
# sim_vals = kernel.evaluate(values)
# # ^this is equivalent to an unscaled pdf as well, so
# sim_pdf = sim_vals / np.sum(sim_vals)

# let's also produce a synthetic set to work backwards to uncertainty:
random_vals_from_kernel = kernel.resample(10000)
fitstats = {}
for n, i in enumerate(['R', 'S', 'F']):
    vals, bins, _ = plt.hist(random_vals_from_kernel[n], bins='auto',
                             cumulative=True, density=True)
    vals = np.insert(vals, 0, 0)
    fitstats[i] = np.interp([0.025, 0.16, 0.5, 0.84, 0.975], vals, bins)

# now, bootstrap the model with the random_vals_from_kernel.
numbugs = []
numbugs_t0 = []
for r, s, f in random_vals_from_kernel.T:
    Fdist = expon(scale=1./f)
    P_flessr = Fdist.cdf(1./r)
    nb = np.log(0.5)/np.log(1. - P_flessr)
    s0 = geom(s).mean() - 1
    if not np.isnan(s0):
        numbugs_t0.append(s0)
    if not np.isnan(nb) and nb >= 0.:
        numbugs.append(nb)
# -> numbugs and numbugs_t0 now contain the distribs of the poss values
print('For 2sd, 1sd, mean, 2sd, 2sd:')
print('Carried bug uncertanties:',
      np.percentile(numbugs, 2.5),
      np.percentile(numbugs, 16.),
      np.percentile(numbugs, 50.),
      np.percentile(numbugs, 84.),
      np.percentile(numbugs, 97.5))
print('Starting bug uncertanties:',
      np.percentile(numbugs_t0, 2.5),
      np.percentile(numbugs_t0, 16.),
      np.percentile(numbugs_t0, 50.),
      np.percentile(numbugs_t0, 84.),
      np.percentile(numbugs_t0, 97.5))

# load the real data
real_bfr = np.loadtxt(
    os.path.join(DATA_DIR,
                 'all_real_data_bug_find_rate.txt')
)
real_commits = np.loadtxt(
    os.path.join(DATA_DIR,
                 'all_real_data_total_commits.txt')
)
real_bins, real_bin_counts, real_binned_bfr = create_bins(
    BINNING_SCALE, real_commits, real_bfr
)
real_bins_std, real_bin_counts_std, real_binned_bfr_std = create_bins(
    BINNING_SCALE_STD, real_commits, real_bfr, method='std'
)

# Now, build the model runs. Take the 1st 10 poss realisations.
print('Beginning resimulation...')
for i in range(10):
    r, s, f = random_vals_from_kernel[:, i]
    # launch the simulation
    avg_sim_bin_bfr = np.zeros_like(real_binned_bfr, dtype=float)
    avg_sim_bin_std = np.zeros_like(real_binned_bfr_std, dtype=float)
    for i in range(REPEATS):
        num_bugs, bug_rate, num_commits = \
            run_with_exponential_num_bugs_floats_in(
                r, s, f, num_realisations='from_data', stochastic=True
            )
        bins, bin_counts, sim_bin_bfr = create_bins(
            BINNING_SCALE, num_commits, bug_rate
        )
        bins_std, bin_counts_std, sim_bin_bfr_std = create_bins(
            BINNING_SCALE_STD, num_commits, bug_rate, method='std'
        )
        avg_sim_bin_bfr += sim_bin_bfr / float(REPEATS)
        avg_sim_bin_std += sim_bin_bfr_std / float(REPEATS)
    plot((bins[1:]+bins[:-1])/2., avg_sim_bin_bfr, 'b')

plot((real_bins[1:]+bins[:-1])/2., real_binned_bfr, 'k')

# add the best fit scenario as well:
r, s, f = np.median(random_vals_from_kernel, axis=1)
# launch the simulation
avg_sim_bin_bfr = np.zeros_like(real_binned_bfr, dtype=float)
avg_sim_bin_std = np.zeros_like(real_binned_bfr_std, dtype=float)
for i in range(REPEATS):
    num_bugs, bug_rate, num_commits = \
        run_with_exponential_num_bugs_floats_in(
            r, s, f, num_realisations='from_data', stochastic=True
        )
    bins, bin_counts, sim_bin_bfr = create_bins(
        BINNING_SCALE, num_commits, bug_rate
    )
    bins_std, bin_counts_std, sim_bin_bfr_std = create_bins(
        BINNING_SCALE_STD, num_commits, bug_rate, method='std'
    )
    avg_sim_bin_bfr += sim_bin_bfr / float(REPEATS)
    avg_sim_bin_std += sim_bin_bfr_std / float(REPEATS)
plot((bins[1:]+bins[:-1])/2., avg_sim_bin_bfr, 'g')
