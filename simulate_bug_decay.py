import numpy as np
import matplotlib.pyplot as plt
from bisect import insort

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
                         total_number_of_bugs_to_find, event_stack,
                         times_of_bug_finds):
    """
    Move forward through an event_stack from time current_time to time
    bug_creation_time, finding bugs in the event_stack and recording them
    in durations_between_bug_finds as necessary. If enough bugs have been
    found before we reach bug_creation_time, returns a ValueError.
    """
    while (current_time + event_stack.time_until_next_bug(current_time)
           < bug_creation_time):
        current_time += event_stack.time_until_next_bug(current_time)
        times_of_bug_finds.append(event_stack.advance())
        if len(times_of_bug_finds) == total_number_of_bugs_to_find:
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
        Raises an IndexError if the event stack lacks any bugs
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

if __name__ == "__main__":
    total_number_of_bugs_to_find = 250
    bug_lifetime_parameter = 0.0003
    # ^dropping this moves us away from a single production line.
    # We drop back from it - but staying parallel - as blp < ~0.01
    # We start to lose parallelism as blp < 0.0005, assoc. w a slow period
    # at the start of the debugging, i.e., bugs need time to build up.
    # In these cases, we are just delayed in approaching the (still identical)
    # background rate.
    # (I don't think our repos are commonly in this zone)
    number_of_starting_bugs = 50
    # The more bugs there are, the more pronounced the kinking the accumulation
    variable_lifetime = False
    times_of_bug_finds = []
    estack = event_stack()
    current_time = 0.

    # set up the initial bugs:
    for i in range(number_of_starting_bugs):
        a_bug = bug(bug_lifetime_parameter, current_time,
                    variable_lifetime=variable_lifetime)
        print("Adding bug, lifetime:", a_bug.lifetime)
        estack.add_a_bug(a_bug)
    # now run the model
    for time_from_now_to_bug_creation in generate_bug():
        # prepare the next bug to be added:
        bug_creation_time = current_time + time_from_now_to_bug_creation
        a_bug = bug(bug_lifetime_parameter, bug_creation_time,
                    variable_lifetime=variable_lifetime)
        # add it
        estack.add_a_bug(a_bug)
        # now advance, finding bugs, until we need to generate a new bug:
        try:
            advance_finding_bugs(current_time, bug_creation_time,
                                 total_number_of_bugs_to_find, estack,
                                 times_of_bug_finds)
        except ValueError:
            break
        # Move through rest of tstep to the point where the bug is created:
        current_time = bug_creation_time


    plt.figure(1)
    plt.plot(times_of_bug_finds, list(range(len(times_of_bug_finds))))
    plt.ylabel('Total number of bugs')
    plt.xlabel('Time found')
    plt.figure(2)
    plt.hist(np.diff(times_of_bug_finds), bins='auto')
    plt.xlabel('Interval between bug finds')
    plt.ylabel('Number of occurrences')
