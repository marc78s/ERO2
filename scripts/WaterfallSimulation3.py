import pandas as pd

from Utils import Utils
from WaterfallSimulation2 import waterfall_simulation_2


class waterfall_simulation_3(waterfall_simulation_2):
    """
    A class representing the waterfall tandem queue of part 2 (M/M/K/ks followed by M/M/1/kf) with a random
    back-up of results.

    Everything is inherited from waterfall_simulation_2 (same parameters, same conventions : ks and kf are total
    capacities, servers included), except file 2. Each result leaving file 1 is backed up with probability p.
    A backed up result finding file 2 full is not lost : it waits in the back-up, of infinite capacity, and
    enters file 2 as soon as a place is free, in FIFO order. A result which is not backed up and finds file 2
    full is lost (blank page). With p = 0 the system is the one of part 2.

    The time spent in the back-up is counted in t_waiting2. tops2 and tops have two more boolean columns :
    backed_up (the result was backed up) and rescued (the result found file 2 full and was kept by the back-up).

    Additional attribute :

        p (float) : probability of backing up a result.
    """


    def __init__(self, lamda, mu1, K, mu2, ks, kf, p, gen, test_z=100, service_law1=None, service_law2=None) :
        """
        Initializes a new instance of the class waterfall_simulation_3.

        Args :
            lamda, mu1, K, mu2, ks, kf, gen, test_z, service_law1, service_law2 : see waterfall_simulation_2.
            p : probability of backing up a result, between 0 and 1.
        """
        if not 0 <= p <= 1:
            raise ValueError("p is a probability, it must be between 0 and 1")

        super().__init__(lamda, mu1, K, mu2, ks, kf, gen, test_z, service_law1, service_law2)
        self.p = p

    def run_mm1(self, t_arrivals=None):
        """
        Simulates file 2 alone with its back-up. Replaces the method of waterfall_simulation_2, which run()
        calls.

        Fills the attribute tops2.

        Args :
            t_arrivals : series of queue arrival times, indexed by agent. If None, arrivals follow a
                         Poisson process of intensity lamda.
        """
        if t_arrivals is None:
            t_arrivals = Utils.poisson_arrivals(self.gen, self.lamda, self.t_size)

        self.tops = None
        self.tops2 = Utils.simulate_queue(self.gen, t_arrivals, self.mu2, 1, self.kf - 1,
                                          p_backup=self.p, service_law=self.service_law2)
        if self.p == 0:  # Utils only adds the columns when there is a back-up
            self.tops2['backed_up'] = False
            self.tops2['rescued'] = False

    def stats(self, file=None):
        """
        Same as waterfall_simulation_2.stats, with the empirical variance of sojourn time (var_sojourn_time).
        """
        return Utils.stats(self.tops1, self.tops2, self.tops, file, variance=True)

    def rescued_rate(self):
        """
        Computes the proportion of results kept by the back-up, among the results that left file 1 : without
        the back-up, these results would have been blank pages.

        Returns :
            The proportion of rescued results, as a float.
        """
        if self.tops2 is None:
            raise RuntimeError("call run() first")
        return self.tops2['rescued'].mean()

    def backup_counts(self, t_intervals):
        """
        Computes number of results in back-up at all given times.

        Args :
            t_intervals : size of time laps between two successive tops.

        Returns :
            A series indexed by time laps of size t_intervals : the results waiting beyond the kf - 1 waiting
            places of file 2.
        """
        return (self.counts(t_intervals, file=2)['ag_in_queue'] - (self.kf - 1)).clip(lower=0)

    def sojourn_by_group(self):
        """
        Computes mean and empirical variance of sojourn time, separately for the results rescued by the back-up
        and for the other delivered results.

        Returns :
            A dataframe indexed by group (rescued, others), with columns count, mean_sojourn_time and
            var_sojourn_time.
        """
        if self.tops is None:
            raise RuntimeError("call run() first")
        delivered = self.tops.dropna(subset=['t_depart_sys2'])
        groups = {'rescued': delivered[delivered['rescued'].astype(bool)],
                  'others': delivered[~delivered['rescued'].astype(bool)]}
        return pd.DataFrame({
            'count': {g: len(d) for g, d in groups.items()},
            'mean_sojourn_time': {g: d['t_sojourn'].mean() for g, d in groups.items()},
            'var_sojourn_time': {g: d['t_sojourn'].var() for g, d in groups.items()},
        })
