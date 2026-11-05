from scripts.Utils import Utils


class waterfall_simulation_1():
    """
    A class representing the waterfall tandem queue : M/M/K (test-suite) followed by M/M/1 (front).

    Each push enters file 1, is served by one of the K servers, then its result enters file 2 and
    is sent to the front by the single server. Both queues are FIFO with infinite capacity, so no
    agent is ever lost.

    Attributes :

        lamda (float)      : parameter of exponential law corresponding to interarrival times.
        mu1 (float)        : parameter of exponential law corresponding to service times of file 1.
        K (int)            : number of servers of file 1.
        mu2 (float)        : parameter of exponential law corresponding to service times of file 2.
        gen (np.random)    : a random generator
        t_size (int)       : number of arrivals simulated.
        tops1              : dataframe of file 1 times for each agent, filled by run_mmk.
        tops2              : dataframe of file 2 times for each agent, filled by run_mm1.
        tops               : dataframe of both files times for each agent, filled by run.
    """


    def __init__(self, lamda, mu1, K, mu2, gen, test_z=100) :
        """
        Initializes a new instance of the class waterfall_simulation_1.

        Args :
            lamda  : parameter of exponential law corresponding to interarrival times.
            mu1    : parameter of exponential law corresponding to service times of file 1.
            K      : number of servers of file 1.
            mu2    : parameter of exponential law corresponding to service times of file 2.
            gen    : a random generator
            test_z : test size.
        """

        self.lamda = lamda
        self.mu1 = mu1
        self.K = K
        self.mu2 = mu2
        self.gen = gen
        self.t_size = test_z

        self.tops1 = None
        self.tops2 = None
        self.tops = None

    def run_mmk(self):
        """
        Simulates file 1 alone, an M/M/K queue fed by a Poisson process of intensity lamda.

        Fills the attribute tops1.
        """
        self.tops = None
        t_arrivals = Utils.poisson_arrivals(self.gen, self.lamda, self.t_size)
        self.tops1 = Utils.simulate_queue(self.gen, t_arrivals, self.mu1, self.K)

    def run_mm1(self, t_arrivals=None):
        """
        Simulates file 2 alone, an M/M/1 queue.

        Fills the attribute tops2.

        Args :
            t_arrivals : series of queue arrival times, indexed by agent. If None, arrivals follow a
                         Poisson process of intensity lamda.
        """
        if t_arrivals is None:
            t_arrivals = Utils.poisson_arrivals(self.gen, self.lamda, self.t_size)

        self.tops = None
        self.tops2 = Utils.simulate_queue(self.gen, t_arrivals, self.mu2, 1)

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
        Computes number of agents in system at all given times.

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
        Computes statistics of a current simulation of queue.

        Args :
            file : 1 or 2 for a single file, None for the whole system.

        Returns:
            A dataframe indexed by standard statistics of interest mean sojourn time (mean_sojourn_time),
            mean waiting time (mean_waiting_time) and mean service time (mean_service_time).
        """
        return Utils.stats(self.tops1, self.tops2, self.tops, file)
