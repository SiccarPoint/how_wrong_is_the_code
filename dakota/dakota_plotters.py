import pandas as pd
from plotnine import *
from matplotlib.pyplot import plot, figure, show

DATFILE = 'dakota_gridtraverse3.dat'
INPARAMS = ["R", "S", "F"]
OUTPARAMS = ["rmse","diff_std_at_low_commits"]

df = pd.read_csv(DATFILE, engine='python', delim_whitespace=True)
df = df.set_index(OUTPARAMS).drop(columns=["interface"])

(
    ggplot(df.reset_index(),
           aes(x=INPARAMS[0], y=INPARAMS[2], color=OUTPARAMS[0]))
    + geom_point()
    + scale_color_continuous(limits=(0., 0.5))
)
