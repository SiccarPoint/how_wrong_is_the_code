import numpy as np
import matplotlib.pyplot as plt
import pymc3 as pm
import theano.tensor as T
from theano import function
from theano.tensor.shared_randomstreams import RandomStreams
from bisect import insort
from scipy.stats import geom
from utils import moving_average

SEED = np.random.randint(10000000)

# assumption 1: bugs have a representative lifetime, measured somehow in
# person-hrs spent on the code. i.e., bugs decay against some timescale.
# assumption 2: bugs differ in how hard they are to spot. i.e., the decay
# chance for our bugs is also exponentially distributed.
# assumption 3: there is some fraction of changes to the code that have bugs

def generate_bug(mean=22., std=6.):
    """Time to generate the next bug. Normally distributed about a mean.
    Defaults are for physics. Creates infinite series.
    """
    while 1:
        ivl = np.clip(np.random.normal(loc=mean, scale=std),
                      a_min=0., a_max=None)
        yield ivl


def advance_finding_bugs(current_time, bug_creation_time,
                         max_total_number_of_bugs_to_find,
                         event_stack,
                         times_of_bug_finds):
    """
    Move forward through an event_stack from time current_time to time
    bug_creation_time, finding bugs in the event_stack and recording them
    in durations_between_bug_finds as necessary. If enough bugs have been
    found before we reach bug_creation_time, returns a ValueError.

    Returns an IndexError if there are no bugs in the stack when called.
    """
    while (current_time + event_stack.time_until_next_bug(current_time)
           < bug_creation_time):
        current_time += event_stack.time_until_next_bug(current_time)
        times_of_bug_finds.append(event_stack.advance())
        if len(times_of_bug_finds) == max_total_number_of_bugs_to_find:
            raise ValueError


class bug():
    def __init__(self, lifetime_parameter, creation_time,
                 variable_lifetime=True):
        """
        Parameters
        ----------
        lifetime_parameter : float
            If variable_lifetime, the decay constant of the exponential
            distribution that describes the actual decay constant of this bug
            (i.e., sets this bug's difficulty of finding).
            If not variable_lifetime, this is the decay constant of the bug.
        creation_time : float
            Current clock time.
        variable_lifetime : bool
            If True (default), the decay parameter of this bug is itself
            exponentially distributed according to lifetime_parameter.
            If False, the decay parameter is fixed.
        """
        for ip in (lifetime_parameter, creation_time):
            assert type(ip) in (np.float64, float, int)
        assert type(variable_lifetime) is bool
        self._lifetime = None
        self._decay_constant = None
        self._decay_time = None
        self._lifetime_parameter = lifetime_parameter
        self._creation_time = creation_time
        if variable_lifetime:
            self.assign_decay_constant(lifetime_parameter)
        else:
            self._decay_constant = lifetime_parameter
        self.assign_bug_lifetime(self._decay_constant)
        self._decay_time = self.calc_time_of_decay(creation_time)

    def assign_decay_constant(self, lifetime_parameter):
        self._decay_constant = 1. / np.random.exponential(
            scale=1./lifetime_parameter
        )

    def assign_bug_lifetime(self, decay_constant):
        self._lifetime = np.random.exponential(scale=1./decay_constant)

    def calc_time_of_decay(self, creation_time):
        return creation_time + self._lifetime

    @property
    def decay_time(self):
        """The scheduled time to be found.
        """
        return self._decay_time

    @property
    def lifetime(self):
        """The scheduled interval until the bug is found, from creation.
        """
        return self._lifetime


class event_stack():
    def __init__(self):
        self._events = []
        # a list of absolute times of known scheduled bug find events
        self._time_of_last_bug = None

    def add_a_bug(self, bug):
        assert (self._time_of_last_bug is None
                or bug.decay_time > self._time_of_last_bug)
        insort(self._events, bug.decay_time)

    def time_until_next_bug(self, current_time):
        """
        Raises an IndexError if the event stack lacks any bugs.

        Returns
        -------
        elapsing_time : float
            The time interval the model will advance to get to the next
            currently planned bug find event.
        """
        time_at_next_bug = self._events[0] # IndexError if empty
        elapsing_time = time_at_next_bug - current_time
        return elapsing_time

    def advance(self):
        """Removes the next event from the stack and returns its time.
        """
        return self._events.pop(0)

    @property
    def events(self):
        return self._events


def run_a_model(max_number_of_bugs_to_find, max_number_of_commits_permissable,
                bug_lifetime_parameter,
                generate_bug_params, number_of_starting_bugs,
                plot_figs):
    """
    Here, the funcion runs until either the bug count reaches
    max_number_of_bugs_to_find, or the "time" exceeds
    max_number_of_commits_permissable.
    """
    #total_number_of_bugs_to_find = 250
    #bug_lifetime_parameter = 0.0001
    # ^dropping this moves us away from a single production line.
    # We drop back from it - but staying parallel - as blp < ~0.01
    # We start to lose parallelism as blp < 0.0005, assoc. w a slow period
    # at the start of the debugging, i.e., bugs need time to build up.
    # In these cases, we are just delayed in approaching the (still identical)
    # background rate.
    # (I don't think our repos are commonly in this zone)
    #number_of_starting_bugs = 50
    # The more bugs there are, the more pronounced the kinking the accumulation
    variable_lifetime = False
    times_of_bug_finds = []
    estack = event_stack()
    current_time = 0.
    time_is_exceeded = False

    # set up the initial bugs:
    for i in range(number_of_starting_bugs):
        a_bug = bug(bug_lifetime_parameter, current_time,
                    variable_lifetime=variable_lifetime)
        # print("Adding bug, lifetime:", a_bug.lifetime)
        estack.add_a_bug(a_bug)
    # now run the model
    for time_from_now_to_bug_creation in generate_bug(generate_bug_params[0],
                                                      generate_bug_params[1]):
        # prepare the next bug to be added:
        bug_creation_time = current_time + time_from_now_to_bug_creation
        if bug_creation_time > max_number_of_commits_permissable:
            bug_creation_time = max_number_of_commits_permissable
            time_is_exceeded = True
        else:
            a_bug = bug(bug_lifetime_parameter, bug_creation_time,
                        variable_lifetime=variable_lifetime)
            # add it
            estack.add_a_bug(a_bug)
        # now advance, finding bugs, until we need to generate a new bug:
        try:
            advance_finding_bugs(current_time, bug_creation_time,
                                 max_number_of_bugs_to_find,
                                 estack,
                                 times_of_bug_finds)
        except (ValueError, IndexError):
            # index error in special case we deplete the bugs just as we
            # are supposed to terminate
            break
        if time_is_exceeded:
            break
        # Move through rest of tstep to the point where the bug is created:
        current_time = bug_creation_time

    if plot_figs:
        plt.figure('Bugs vs commits')
        plt.plot(times_of_bug_finds, list(range(len(times_of_bug_finds))))
        plt.ylabel('Total number of bugs')
        plt.xlabel('Time found')
        plt.figure('Wait times')
        plt.hist(np.diff(times_of_bug_finds), bins='auto')
        plt.xlabel('Interval between bug finds')
        plt.ylabel('Number of occurrences')

    return times_of_bug_finds


def run_with_fixed_num_bugs(rates, start_bugs, num_realisations,
                            generate_bug_params, plot_figs=False):
    doi_bug_commit_distn = np.loadtxt('doiorg_total_commits_for_each_repo.txt')
    out_dict = {}
    for rate in rates:
        out_dict[rate] = {}
        for num_start_bugs in start_bugs:
            out_dict[rate][num_start_bugs] = {}
            out_dict[rate][num_start_bugs]['num_commits'] = []
            out_dict[rate][num_start_bugs]['bug_rate'] = []
            for i in range(num_realisations):
                # draw a plausible repo length:
                repo_len = np.random.choice(doi_bug_commit_distn)
                # repo_len = 1000
                times_of_bug_finds = run_a_model(
                    10000, repo_len, rate, generate_bug_params, num_start_bugs,
                    plot_figs
                )
                # (10,3) very approx for doi.org
                # really big poss number of bugs to get the termination at
                # time not bug count
                # An early cutoff before repo len without reaching num_bugs
                # indicates the model did not find more bugs in the remaining
                # interval

                number_caught = len(times_of_bug_finds)
                bug_rate = number_caught / repo_len
                out_dict[rate][num_start_bugs]['num_commits'].append(repo_len)
                out_dict[rate][num_start_bugs]['bug_rate'].append(bug_rate)
    return out_dict


def run_with_exponential_num_bugs(rates, start_bug_exp_scales,
                                  num_realisations, generate_bug_params,
                                  plot_figs=False):
    doi_bug_commit_distn = np.loadtxt('doiorg_total_commits_for_each_repo.txt')
    out_dict = {}
    for rate in rates:
        out_dict[rate] = {}
        for exp_scale in start_bug_exp_scales:
            out_dict[rate][exp_scale] = {}
            out_dict[rate][exp_scale]['num_commits'] = []
            out_dict[rate][exp_scale]['bug_rate'] = []
            start_bugs = np.random.geometric(exp_scale,
                                             num_realisations) - 1
            # geometric -> exponential discrete equivalent
            # -1 to start from 0 not 1
            for num_start_bugs in start_bugs:
                # draw a plausible repo length:
                repo_len = np.random.choice(doi_bug_commit_distn)
                # repo_len = 1000
                times_of_bug_finds = run_a_model(
                    10000, repo_len, rate, generate_bug_params, num_start_bugs,
                    plot_figs
                )
                # (10,3) very approx for doi.org
                # really big poss number of bugs to get the termination at
                # time not bug count
                # An early cutoff before repo len without reaching num_bugs
                # indicates the model did not find more bugs in the remaining
                # interval

                number_caught = len(times_of_bug_finds)
                bug_rate = number_caught / repo_len
                out_dict[rate][exp_scale]['num_commits'].append(repo_len)
                out_dict[rate][exp_scale]['bug_rate'].append(bug_rate)
    return out_dict


def run_with_exponential_num_bugs_floats_in(r, s, b, num_realisations):
    doi_bug_commit_distn = np.loadtxt('doiorg_total_commits_for_each_repo.txt')
    start_bugs = np.random.geometric(s, num_realisations) - 1
    repo_len = np.random.choice(doi_bug_commit_distn)
    nums_caught = []
    bug_rates = []
    for num_start_bugs in start_bugs:
        times_of_bug_finds = run_a_model(
            10000, repo_len, r, (b, 1.), num_start_bugs, plot_figs=False
        )
        number_caught = len(times_of_bug_finds)
        bug_rate = number_caught / repo_len
        nums_caught.append(number_caught)
        bug_rates.append(bug_rate)
    return np.array(nums_caught), np.array(bug_rates)


def run_exp_three_times_and_bin(theta, x, n=1000):
    """
    Run the exponential form of the driver three times and report the binned
    answer. This is in this form to enamble pymc3 functionality.

    Parameters
    ----------
    theta :
        iterable of the driving params, (r, s, b)
    x :
        the bin intervals, i.e., the dependent variable
    n =
    """
    r, s, b = theta
    bin_vals = [0., ] * (len(x) - 1)
    bin_vals = np.array(bin_vals)
    for i in range(3):
        num_commits, bug_rate = run_with_exponential_num_bugs_floats_in(
            r, s, b, n
        )
        total_commits_order = np.argsort(num_commits)
        total_commits_IN_order = num_commits[total_commits_order]
        bug_find_rate_ordered = bug_rate[total_commits_order]
        bbase_index = 0
        for en, btop in enumerate(x[1:]):
            btop_index = np.searchsorted(total_commits_IN_order, btop, 'right')
            bin_vals[en] += np.mean(
                bug_find_rate_ordered[bbase_index:btop_index]
            )
            bbase_index = btop_index
    bin_vals /= 3.
    return bin_vals


def my_loglike(theta, x, data, sigma):
    """
    A Gaussian log-likelihood function for a model with parameters given in
    theta.
    """
    sim_bins = run_exp_three_times_and_bin(theta, x)

    return -(0.5 / sigma ** 2) * np.sum((data - sim_bins) ** 2)


class theano_Op_wrapper(T.Op):
    """
    Specify what type of object will be passed and returned to the Op when it is
    called. In our case we will be passing it a vector of values (the parameters
    that define our model) and returning an array of the vals to compare.
    """
    itypes = [T.dvector] # expects a vector of parameter values when called
    otypes = [T.dscalar] # outputs a single scalar value

    def __init__(self, loglike, data, x, sigma):
        """
        Initialise the Op with various things that our simulator function
        requires.

        Parameters
        ----------
        loglike :
            The log-likelihood driver func
        data :
            The true data to match to
        x :
            A dummy
        sigma :
            The noise std that our function requires
        """
        self._likelihood = loglike
        self._data = data
        self._x = x
        self._sigma = sigma

    def perform(self, node, inputs, outputs):
        # the method that is used when calling the Op
        theta, = inputs  # this will contain my variables

        # call the func
        log1 = self._likelihood(theta, self._x, self._data, self._sigma)

        outputs[0][0] = np.array(log1)


def mcmc_fitter(n_samples=5, n_burn=2):
    """
    Run the MCMC model. Defaults represent a test run.

    Parameters:
    -----------
    n_samples : int
        number of draws from the distribution
    n_burn : int
        number of "burn-in" points (which we'll discard)
    """
    # attempts an MCMC fit to the data. Hard part is a sensible fit model.
    # Only sensible one is a binned model that lets us overcome the noise
    # near 0. Adding mean and std may help?
    # breaks at [0., 1., 2., 3., 4., 5., 7., 10., 20., 50., 100., 200., 1000000.]
    # inspired by https://github.com/WillKoehrsen/ai-projects/blob/master/markov_chain_monte_carlo/markov_chain_monte_carlo.ipynb
    # follows the black box approach of https://docs.pymc.io/notebooks/blackbox_external_likelihood.html
    real_data = np.loadtxt('real_data_count.txt')
    bin_intervals = np.loadtxt('real_data_bin_intervals.txt')
    sigma = 5.  # std of noise...??

    # make the black box Op model
    log1 = theano_Op_wrapper(my_loglike, real_data, bin_intervals, sigma)

    with pm.Model() as model:
        r = pm.Uniform('r', 0.00001, 0.1, testval=0.01)
        s = pm.Uniform('s', 0.001, 0.5, testval=0.1)
        b = pm.Uniform('b', 0.001, 30., testval=5.)

        # convert these to a tensor variable
        theta = T.as_tensor_variable([r, s, b])

        # use a DensityDist (use a lambda func to call the Op)
        pm.DensityDist('likelihood', lambda v: log1(v),
                       observed={'v': theta})

        trace = pm.sample(n_samples, tune=n_burn, discard_tuned_samples=True,
                          njobs=4)

    return trace


if __name__ == "__main__":
    run_type = 'exp'  # {'fixed', 'exp'}
    rates = (0.01, )# (0.0003, 0.001, 0.003)
    if run_type == 'fixed':
        start_bugs = (1000, )  # (0, 50, 250)
        out_dict = run_with_fixed_num_bugs(rates, start_bugs, 1000, (10., 3.)
                                           plot_figs=True)
        dict_keys = start_bugs
    elif run_type == 'exp':
        exp_scales = (0.2, )#(0.1, 0.2)
        # rate = 0.001 scale ~0.1-0.2 gives interesting responses around the
        # sweet spot where no sensitivity transitions to fits that are
        # sensitive but poor - but this param combo cannot give saturation by
        # ~300 commits. Dropping helps, but then we lose good fit
        # v little sensitivity to scale at the high end
        # i.e., there needs to be a wide spread in poss number of bugs,
        # i.e., up to hundreds of bugs, to get interesting responses
        # by 0.1 we are dealing mostly w 10s of bugs only, depleted too fast
        # in ALL runs.
        out_dict = run_with_exponential_num_bugs(rates, exp_scales, 1000,
                                                 (7.5, 1.), plot_figs=True)
        # can produce a pretty good version of the key plot with r=0.003,
        # s=0.2, b=(10., 3.)... but it "levels out" at 0.04, not 0.1.
        # So, drop b to get bigger number and then re-tune the others?
        # r=0.01, s=0.2, b=(7.5, 1.) looks pretty great
        # s~0.2 is ~10 bugs on commit
        # Under these terms we have good s sensitivity! big s, i.e., no bugs
        # at commit, creates a poor, linear fit with no rollover
        # By s~0.2, we get a good roll, a tendency to bfr~0.1 at hi commits,
        # and the noise close to 0 commits stays below 0.1.
        # By s~0.1, retain nice rollover, stable bfr goes up slightly (0.12?)
        # but the rollover happens sooner and the noise at zero is much bigger
        # (0->0.2).
        # Bottom line is that there is very nice sensitivity to all params
        # available here
        # We could also add the CDF to get another constraint
        dict_keys = exp_scales
    else:
        raise NameError('run_type not recognised')

    # now mock up figures
    for rate in rates:
        for k in dict_keys:
            runname = str(rate) + '_' + str(k)
            plt.figure(runname)
            num_commits = np.array(
                out_dict[rate][k]['num_commits']
            )
            bug_rate = np.array(
                out_dict[rate][k]['bug_rate']
            )
            plt.plot(num_commits, bug_rate, 'x')
            plt.xlabel('Commits in repo')
            plt.ylabel('Apparent bug find rate')

            total_commits_order = np.argsort(num_commits)
            total_commits_IN_order = num_commits[total_commits_order]
            bug_find_rate_ordered = bug_rate[total_commits_order]
            bug_find_rate_moving_avg = moving_average(bug_find_rate_ordered,
                                                      n=20)
            # # also a moving average excluding the rate = 0 cases
            # # (=> is it just them causing the trend?)
            # # plotting will show the trend is still there without zeros
            # rate_not_zero = np.logical_not(np.isclose(bug_find_rate_ordered,
            #                                           0.))
            # bug_find_rate_moving_avg_no_zeros = moving_average(
            #     bug_find_rate_ordered[rate_not_zero], n=20
            # )
            # commits_more_than_1 = total_commits_IN_order > 1
            # bug_find_rate_moving_avg_not1commit = moving_average(
            #     bug_find_rate_ordered[commits_more_than_1], n=20
            # )
            plt.plot(total_commits_IN_order[:-19],
                     bug_find_rate_moving_avg, '-')

            plt.figure('all_runs')
            # plt.plot(num_commits, bug_rate, 'x')
            plt.plot(total_commits_IN_order[:-19],
                     bug_find_rate_moving_avg, '-', label=runname)
            plt.xlabel('Commits in repo')
            plt.ylabel('Apparent bug find rate')
            plt.legend()
