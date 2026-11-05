from scripts.Utils import Utils


class waterfall_simulation_2():
    """
    A class representing the waterfall tandem queue with finite queues : M/M/K/(K+ks) followed by M/M/1/(kf+1).

    A push arriving while the ks places of file 1 are taken is refused (error message). A result arriving
    while the kf places of file 2 are taken is lost (blank page). A refused agent keeps its row in the
    dataframes, with NaN for the times it never reached, so statistics only cover the agents that were served.

    Attributes :

        lamda (float)      : parameter of exponential law corresponding to interarrival times.
        mu1 (float)        : parameter of exponential law corresponding to service times of file 1.
        K (int)            : number of servers of file 1.
        mu2 (float)        : parameter of exponential law corresponding to service times of file 2.
        ks (int)           : capacity of file 1, servers excluded.
        kf (int)           : capacity of file 2, server excluded.
        gen (np.random)    : a random generator
        t_size (int)       : number of arrivals simulated.
        tops1              : dataframe of file 1 times for each agent, filled by run_mmk.
        tops2              : dataframe of file 2 times for each agent, filled by run_mm1.
        tops               : dataframe of both files times for each agent, filled by run.
    """


    def __init__(self, lamda, mu1, K, mu2, ks, kf, gen, test_z=100) :
        """
        Initializes a new instance of the class waterfall_simulation_2.

        Args :
            lamda  : parameter of exponential law corresponding to interarrival times.
            mu1    : parameter of exponential law corresponding to service times of file 1.
            K      : number of servers of file 1.
            mu2    : parameter of exponential law corresponding to service times of file 2.
            ks     : capacity of file 1, servers excluded.
            kf     : capacity of file 2, server excluded.
            gen    : a random generator
            test_z : test size.
        """

        self.lamda = lamda
        self.mu1 = mu1
        self.K = K
        self.mu2 = mu2
        self.ks = ks
        self.kf = kf
        self.gen = gen
        self.t_size = test_z

        self.tops1 = None
        self.tops2 = None
        self.tops = None

    def run_mmk(self):
        """
        Simulates file 1 alone, an M/M/K/(K+ks) queue fed by a Poisson process of intensity lamda.

        Fills the attribute tops1.
        """
        self.tops = None
        t_arrivals = Utils.poisson_arrivals(self.gen, self.lamda, self.t_size)
        self.tops1 = Utils.simulate_queue(self.gen, t_arrivals, self.mu1, self.K, self.ks)

    def run_mm1(self, t_arrivals=None):
        """
        Simulates file 2 alone, an M/M/1/(kf+1) queue.

        Fills the attribute tops2.

        Args :
            t_arrivals : series of queue arrival times, indexed by agent. If None, arrivals follow a
                         Poisson process of intensity lamda.
        """
        if t_arrivals is None:
            t_arrivals = Utils.poisson_arrivals(self.gen, self.lamda, self.t_size)

        self.tops = None
        self.tops2 = Utils.simulate_queue(self.gen, t_arrivals, self.mu2, 1, self.kf)

    def run(self):
        """
        Simulates the whole system : the departures of file 1 are the arrivals of file 2.

        Fills tops1, tops2 and tops. The dataframe tops has one row per push and contains the columns
        of both files, suffixed by 1 or 2, and the total durations t_sojourn, t_waiting and t_service.
        """
        self.run_mmk()
        self.run_mm1(self.tops1['t_depart_sys'])
        self.tops = Utils.join_files(self.tops1, self.tops2)

    def counts(self, t_intervals, file=None):
        """
        Computes number of agents in system at all given times, refused agents excluded.

        Args :
            t_intervals : size of time laps between two successive tops.
            file        : 1 or 2 for a single file, None for the whole system (both queues, both services).

        Returns :
            A dataframe indexed by time laps of size t_intervals all along total queue simulation time,
            its three columns contain number of agents in system (ag_in_sys), in arrival queue (ag_in_queue) and
            in service (ag_in_service).
        """
        return Utils.counts(self.tops1, self.tops2, self.tops, t_intervals, file)

    def stats(self, file=None):
        """
        Computes statistics of a current simulation of queue, over the agents that were served.

        Args :
            file : 1 or 2 for a single file, None for the whole system.

        Returns:
            A dataframe indexed by standard statistics of interest mean sojourn time (mean_sojourn_time),
            mean waiting time (mean_waiting_time) and mean service time (mean_service_time).
        """
        return Utils.stats(self.tops1, self.tops2, self.tops, file)

    def refusal_rate(self, file=None):
        """
        Computes the proportion of agents refused.

        Args :
            file : 1 for the pushes refused by file 1, 2 for the results refused by file 2 (blank pages,
                   among the results that left file 1), None for the pushes that never get a result,
                   whichever file refused them.

        Returns :
            The proportion of refused agents, as a float.
        """
        return Utils.refusal_rate(self.tops1, self.tops2, self.tops, file)
