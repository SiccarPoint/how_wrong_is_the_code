import numpy as np

# assumption 1: bugs have a representative lifetime, measured somehow in
# person-hrs spent on the code. i.e., bugs decay against some timescale.
# assumption 2: bugs differ in how hard they are to spot. i.e., the decay
# chance for our bugs is also exponentially distributed.
# assumption 3: there is some fraction of changes to the code that have bugs

BUG_GENERATION_RATE = 0.1  # per time

class bug():
    def __init__(self, lifetime_parameter, variable_lifetime=True):
        """
        Parameters
        ----------
        lifetime_parameter : float
            If variable_lifetime, the decay constant of the exponential
            distribution that describes the actual decay constant of this bug
            (i.e., sets this bug's difficulty of finding).
            If not variable_lifetime, this is the decay constant of the bug.
        variable_lifetime : bool
            If True (default), the decay parameter of this bug is itself
            exponentially distributed according to lifetime_parameter.
            If False, the decay parameter is fixed.
        """
        assert type(lifetime_parameter) in (float, int)
        assert type(variable_lifetime) is bool
        self._lifetime = None
        self._decay_constant = None
        self._lifetime_parameter = lifetime_parameter
        if variable_lifetime:
            self.assign_decay_constant(lifetime_parameter)
        else:
            self._decay_constant = lifetime_parameter
        self._lifetime = calc_bug_lifetime(self._decay_constant)

    def assign_decay_constant(self, lifetime_parameter):
        self._decay_constant = np.random.exponential(
            scale=1./lifetime_parameter
        )

    def assign_bug_lifetime(self, decay_constant):
        self._lifetime = np.random.exponential(scale=1./decay_constant)

    def calc_time_of_decay(self, creation_time):
        return creation_time + self._lifetime


class event_stack():
    def __init__(self):
        
