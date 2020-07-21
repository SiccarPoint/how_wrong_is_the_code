import pandas as pd
from plotnine import *
from matplotlib.pyplot import plot, figure, show
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objs as go
from plotly.offline import iplot
from scipy.stats import gaussian_kde, expon, geom
import numpy as np

DATFILE = 'dakota_gridtraverse_N1.dat'
INPARAMS = ["R", "S", "F"]
# OUTPARAMS = ["rmse","diff_std_at_low_commits"]
OUTPARAMS = ["rmse","rmse_std"]

df = pd.read_csv(DATFILE, engine='python', delim_whitespace=True)
df = df.set_index(OUTPARAMS).drop(columns=["interface"])

plot1 = (
    ggplot(df.reset_index(),
           aes(x=INPARAMS[0], y=INPARAMS[1], color=OUTPARAMS[0]))
    + geom_point()
    + scale_color_continuous(limits=(0.02, 0.04))
)

plot2 = (
    ggplot(df.reset_index(),
           aes(x=INPARAMS[0], y=INPARAMS[2], color=OUTPARAMS[0]))
    + geom_point()
    + scale_color_continuous(limits=(0.022, 0.03))
)

plot3 = (
    ggplot(df.reset_index(),
           aes(x=INPARAMS[1], y=INPARAMS[2], color=OUTPARAMS[0]))
    + geom_point()
    + scale_color_continuous(limits=(0.02, 0.04))
)

plot4 = (
    ggplot(df.reset_index(),
           aes(x=INPARAMS[0], y=INPARAMS[1], color=OUTPARAMS[1]))
    + geom_point()
    + scale_color_continuous(limits=(0.02, 0.08))
)

plot5 = (
    ggplot(df.reset_index(),
           aes(x=INPARAMS[0], y=INPARAMS[2], color=OUTPARAMS[1]))
    + geom_point()
    + scale_color_continuous(limits=(0.05, 0.06))
)

plot6 = (
    ggplot(df.reset_index(),
           aes(x=INPARAMS[1], y=INPARAMS[2], color=OUTPARAMS[1]))
    + geom_point()
    + scale_color_continuous(limits=(0.03, 0.06))
)

df = pd.read_csv(DATFILE, engine='python', delim_whitespace=True)
marker = {
    'size': 3,
    'color': df[OUTPARAMS[0]],
    'colorscale': 'Viridis',
    'cmin': 0.0145,
    'cmax': 0.018
}
marker2 = {
    'size': 3,
    'color': df[OUTPARAMS[0]] + df[OUTPARAMS[1]],
    'colorscale': 'Viridis',
    'cmin': 0.047,
    'cmax': 0.055
}
trace7 = go.Scatter3d(
    x=df[INPARAMS[0]], y=df[INPARAMS[1]], z=df[INPARAMS[2]],
    mode='markers', marker=marker2,
)
layout = go.Layout(title='3D RMSE map')
fig = go.Figure(data=[trace7], layout=layout)
iplot(fig)

# This plot demonstrates that there is a zone of values where we can find
# low misfit solutions. Adding in the STD RMSE brings the zone down, but does
# not eliminate it.
# Try a gaussian_kde fit on thresholded dataset to assess this
R = df[INPARAMS[0]]
S = df[INPARAMS[1]]
F = df[INPARAMS[2]]
RMSE = df[OUTPARAMS[0]]
RMSE_TOT = df[OUTPARAMS[0]] + df[OUTPARAMS[1]]
values = np.vstack([R, S, F])
kernel = gaussian_kde(values, weights=1./RMSE)
sim_vals = kernel.evaluate(values)
# ^this is equivalent to an unscaled pdf as well, so
sim_pdf = sim_vals / np.sum(sim_vals)

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

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(R, S, F, c=sim_pdf, marker='o', s=3)
ax.set_xlabel('R')
ax.set_ylabel('S')
ax.set_zlabel('F')
