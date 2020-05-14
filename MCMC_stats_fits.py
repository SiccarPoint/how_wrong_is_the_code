import numpy as np
import matplotlib.pyplot as plt
import pymc3 as pm
import theano.tensor as T
from theano import function
from theano.tensor.shared_randomstreams import RandomStreams
from bisect import insort
from scipy.stats import geom
from utils import moving_average
from find_grads import gradients  # ensure to run python setup.py install
from driver import moving_average

def statistical_model_bugcount():
    """
    Mocks up a purely statistical treatment of the model in the hope pymc3
    can be applied w/o a black box.
    https://docs.pymc.io/notebooks/data_container.html
    """
    # x = ...  # predictor variable; number of commits per repo
    # # priors
    all_real_data = np.loadtxt('all_real_data_num_bugs.txt')
    all_commits = np.loadtxt('all_real_data_total_commits.txt')
    # all_real_data = np.loadtxt('real_data_bin_means.txt')
    # all_commits = np.loadtxt('real_data_bin_intervals.txt')
    # all_commits = (all_commits[:-1] + all_commits[1:])/2
    # at a given ncommits, the observed bugs should be a BINOMIAL distb, as
    # we are counting observations in a yes/no system
    # We are also explicitly dealing with SURVIVAL FUNCS, specifically exp
    # as argued here, but arguably also Weibull with a decreasing chance of
    # bug finding (i.e., superexponential, "Lindy func").
    total_bugs_moving_avg = moving_average(all_real_data, n=20)

    with pm.Model() as model_bugs:
        commits = pm.Data('commits', all_commits)
        # Now set up the crucial unknowns - those for the starting bug popn
        S = pm.Uniform('S', 0.000001, 10., testval=0.1)
        N0 = pm.Exponential('N0', S)
        # R = pm.Normal('R', mu=0.2, sigma=0.05)  # generation rate AKA A
        # F = pm.Normal('F', mu=0.2, sigma=0.05)  # find rate
        R = pm.Uniform('R', 0.000001, 1., testval=0.01)  # generation rate AKA A
        F = pm.Uniform('F', 0.000001, 1., testval=0.01)  # find rate

        # The starting popn decays as a simple exponential, i.e.,
        # N_startsurvivors = N0 * T.exp(-F * commits)
        # which means the bug count goes as
        N_found_startpopn = pm.Deterministic(
            'N_found_startpopn',
            N0.random(size=len(all_commits)) * (1. - T.exp(-F * commits))
        )

        # now, we generate bugs as we go. So,
        #N_added = pm.Deterministic('N_added', R * commits)
        # and we can only find bugs we add, so
        # N_added_found = pm.Deterministic(
        #     'N_added_found', N_added * (1. - T.exp(-F * commits))
        # )
        N_added_found = pm.Deterministic(
            'N_added_found', R * commits * (1. - T.exp(-F * commits))
        )
        # and so the total bugs found:
        N_found_total = pm.Deterministic(
            'N_found_total', (N_found_startpopn + N_added_found) / commits
        )

        # and so, for any given ncommits, we expect a binomial distribution,
        # i.e.,
        bugs_pred = pm.Poisson('bugs_pred', N_found_total,
                               observed=all_real_data)

        trace = pm.sample(10000, tune=2000, cores=4)
