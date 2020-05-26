import pandas as pd
from plotnine import *
from matplotlib.pyplot import plot, figure, show

DATFILE = 'dakota_gridtraverse4b.dat'
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
