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

from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objs as go
from plotly.offline import iplot

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

# Now, build the model runs. Take the 1st 10 poss realisations.
print('Beginning resimulation...')
for j in range(10):
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

# The real dataset has an implicit RMSE. What is it?
# We can use this to threshold our good fits
# Because we only have one realisation of the "real" data, we can't get it
# direct. But, the variability of the model runs that fit well will proxy
# the real behaviour.
# We should do this iteratively really, but to get us going we know the kernel
# approach gets us r=0.095, s=0.641, f=0.041 as medians, and these plot up
# looking decent
# so now we get a bunch of realisations and compare them to their mean:
r, s, f = np.median(random_vals_from_kernel, axis=1)  # as above
realz = 25
avg_sim_bin_bfr = np.zeros((realz, real_binned_bfr.size), dtype=float)
avg_sim_bin_std = np.zeros((realz, real_binned_bfr_std.size), dtype=float)
figure(2)
for j in range(realz):
    # launch the simulation
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
        avg_sim_bin_bfr[j] += sim_bin_bfr / float(REPEATS)
        avg_sim_bin_std[j] += sim_bin_bfr_std / float(REPEATS)
    plot((bins[1:]+bins[:-1])/2., avg_sim_bin_bfr[j], 'b')

mean_avg_sim_bfr = avg_sim_bin_bfr.mean(axis=1)
each_sim_rmse = (
    (avg_sim_bin_bfr.T - mean_avg_sim_bfr).T**2
).mean(axis=1) ** 0.5
mean_sim_rmse = each_sim_rmse.mean()
# This is ~0.02. ~1900 realisations of the param space produce RMSE at least
# this good (of ~32800). So we can threshold against this and THEN do our
# kernel approach, without a weighting.
thresholded_rmse_outputs = RMSE < mean_sim_rmse
low_rmse_combos = (values.T[np.asarray(thresholded_rmse_outputs)]).T
# Don't even need to kernel it! Just bootstrap direct off the subset...

figure(3)
ax = fig.add_subplot(111, projection='3d')
ax.scatter(low_rmse_combos[0], low_rmse_combos[1], low_rmse_combos[2], marker='o', s=3)
ax.set_xlabel('R')
ax.set_ylabel('S')
ax.set_zlabel('F')
# makes the "foot" of decent vals pretty clear.

idx = np.random.randint(low_rmse_combos.shape[1], size=1000)
random_good_vals = low_rmse_combos[:, idx]
# Now, build the model runs. Take the 1st 50 poss realisations.
print('Beginning resimulation, again...')
figure(4)
for j in range(50):
    r, s, f = random_good_vals[:, i]
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
    plot((bins[1:]+bins[:-1])/2., avg_sim_bin_bfr, 'b', alpha=0.1)

plot((real_bins[1:]+bins[:-1])/2., real_binned_bfr, 'k')

# so now again:
print('Uncertainties based on inherent variabilities in best answers:')
numbugs = []
numbugs_t0 = []
for r, s, f in random_good_vals.T:
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

# & for bootstrapping the values we use to derive this:
r, s, f = np.median(random_good_vals, axis=1)
# this changes r, s, f by a small few %, and barely touches n estimates
