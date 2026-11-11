import heapq
from collections import deque

import numpy as np
import pandas as pd


class Utils():
    """
    Functions shared by the classes waterfall_simulation_1, waterfall_simulation_2 and waterfall_simulation_3.

    In the dataframes, a refused agent keeps its row with NaN for the times it never reached, so statistics
    only cover the agents that were served.
    """


    @staticmethod
    def poisson_arrivals(gen, lamda, t_size):
        """
        Returns the absolute arrival times of a Poisson process of intensity lamda, indexed by agent.
        """
        return pd.Series(np.cumsum(gen.exponential(1.0 / lamda, t_size)))

    @staticmethod
    def simulate_queue(gen, t_arrivals, mu, n_servers, capacity=np.inf, p_backup=0.0, service_law=None):
        """
        Simulates a FIFO queue with n_servers identical servers, exponential by default.

        Args :
            gen         : a random generator
            t_arrivals  : series of queue arrival times, indexed by agent. NaN (agents refused upstream) are ignored.
            mu          : service rate, the mean service time is 1/mu.
            n_servers   : number of servers.
            capacity    : number of places in queue, servers excluded.
            p_backup    : probability that an agent is backed up, in which case it is never refused and waits
                          for a place when the queue is full.
            service_law : function (gen, mean, size) -> array of service times. If None, exponential law.

        Returns :
            A dataframe indexed by agent, sorted by queue arrival time, containing queue arrival times
            (t_arval_queue), service arrival times (t_arval_srv), departure times (t_depart_sys),
            sojourn (t_sojourn), waiting (t_waiting) and service (t_service) times. Refused agents have
            NaN everywhere but in t_arval_queue.
        """
        t_arrivals = t_arrivals.dropna().sort_values()
        t_arval_queue = t_arrivals.to_numpy()
        n = len(t_arval_queue)
        if service_law is None:
            services = gen.exponential(1.0 / mu, n)
        else:
            services = service_law(gen, 1.0 / mu, n)
        backed_up = gen.random(n) < p_backup if p_backup > 0 else np.zeros(n, dtype=bool)

        t_arval_srv = np.full(n, np.nan)
        t_depart_sys = np.full(n, np.nan)

        free_at = [0.0] * n_servers
        waiting = deque()

        for i, t in enumerate(t_arval_queue):
            while waiting and waiting[0] <= t:
                waiting.popleft()

            start = max(t, free_at[0])
            if start > t:
                if len(waiting) >= capacity and not backed_up[i]:
                    continue
                waiting.append(start)

            t_arval_srv[i] = start
            t_depart_sys[i] = start + services[i]
            heapq.heapreplace(free_at, t_depart_sys[i])

        tops = pd.DataFrame({
            't_arval_queue': t_arval_queue,
            't_arval_srv': t_arval_srv,
            't_depart_sys': t_depart_sys
        }, index=t_arrivals.index)
        tops.index.name = 'agent'

        tops['t_sojourn'] = tops['t_depart_sys'] - tops['t_arval_queue']
        tops['t_waiting'] = tops['t_arval_srv'] - tops['t_arval_queue']
        tops['t_service'] = tops['t_depart_sys'] - tops['t_arval_srv']
        return tops

    @staticmethod
    def join_files(tops1, tops2):
        """
        Returns one dataframe per push with the columns of both files, suffixed by 1 or 2, and the total
        durations t_sojourn, t_waiting and t_service.
        """
        tops = tops1.join(tops2, lsuffix='1', rsuffix='2')
        tops['t_sojourn'] = tops['t_depart_sys2'] - tops['t_arval_queue1']
        tops['t_waiting'] = tops['t_waiting1'] + tops['t_waiting2']
        tops['t_service'] = tops['t_service1'] + tops['t_service2']
        return tops

    @staticmethod
    def stages(tops1, tops2, tops, file):
        """
        Returns the list of per-file dataframes matching file (1, 2 or None for the whole system),
        without the agents refused by that file.
        """
        if file not in (None, 1, 2):
            raise ValueError("file must be 1, 2 or None")

        stages = {None: [tops1, tops2], 1: [tops1], 2: [tops2]}[file]
        missing = tops is None if file is None else stages[0] is None
        if missing:
            method = {None: 'run', 1: 'run_mmk', 2: 'run_mm1'}[file]
            raise RuntimeError(f"call {method}() first")
        return [stage.dropna(subset=['t_depart_sys']) for stage in stages]

    @staticmethod
    def count_between(t_start, t_end, time_steps):
        """
        Counts, at each time step, the agents such that t_start <= t < t_end.
        """
        started = np.searchsorted(np.sort(t_start), time_steps, side='right')
        ended = np.searchsorted(np.sort(t_end), time_steps, side='right')
        return started - ended

    @staticmethod
    def counts(tops1, tops2, tops, t_intervals, file=None):
        """
        Computes number of agents in system at all given times.

        Args :
            tops1, tops2, tops : dataframes of file 1, file 2 and both files.
            t_intervals        : size of time laps between two successive tops.
            file               : 1 or 2 for a single file, None for the whole system (both queues, both services).

        Returns :
            A dataframe indexed by time laps of size t_intervals all along total queue simulation time,
            its three columns contain number of agents in system (ag_in_sys), in arrival queue (ag_in_queue) and
            in service (ag_in_service).
        """
        stages = Utils.stages(tops1, tops2, tops, file)
        max_time = max(stage['t_depart_sys'].max() for stage in stages)
        time_steps = np.arange(0, max_time + t_intervals, t_intervals)

        ag_in_queue = sum(Utils.count_between(stage['t_arval_queue'], stage['t_arval_srv'], time_steps) for stage in stages)
        ag_in_service = sum(Utils.count_between(stage['t_arval_srv'], stage['t_depart_sys'], time_steps) for stage in stages)

        counts_df = pd.DataFrame({
            'ag_in_sys': ag_in_queue + ag_in_service,
            'ag_in_queue': ag_in_queue,
            'ag_in_service': ag_in_service
        }, index=time_steps)
        counts_df.index.name = 'time'
        return counts_df

    @staticmethod
    def stats(tops1, tops2, tops, file=None, variance=False):
        """
        Computes statistics of a current simulation of queue, over the agents that were served.

        Args :
            tops1, tops2, tops : dataframes of file 1, file 2 and both files.
            file               : 1 or 2 for a single file, None for the whole system.
            variance           : if True, adds the empirical variance of sojourn time.

        Returns:
            A dataframe indexed by standard statistics of interest mean sojourn time (mean_sojourn_time),
            mean waiting time (mean_waiting_time), mean service time (mean_service_time) and, if asked,
            variance of sojourn time (var_sojourn_time).
        """
        stages = Utils.stages(tops1, tops2, tops, file)
        selected = tops if file is None else stages[0]

        stat_values = [
            selected['t_sojourn'].mean(),
            selected['t_waiting'].mean(),
            selected['t_service'].mean()
        ]
        stat_names = ['mean_sojourn_time', 'mean_waiting_time', 'mean_service_time']
        if variance:
            stat_values.append(selected['t_sojourn'].var())
            stat_names.append('var_sojourn_time')
        return pd.DataFrame(stat_values, index=stat_names, columns=['value'])

    @staticmethod
    def refusal_rate(tops1, tops2, tops, file=None):
        """
        Computes the proportion of agents refused.

        Args :
            tops1, tops2, tops : dataframes of file 1, file 2 and both files.
            file               : 1 for the pushes refused by file 1, 2 for the results refused by file 2
                                 (blank pages, among the results that left file 1), None for the pushes
                                 that never get a result, whichever file refused them.

        Returns :
            The proportion of refused agents, as a float.
        """
        Utils.stages(tops1, tops2, tops, file)
        if file is None:
            return tops['t_depart_sys2'].isna().mean()
        return {1: tops1, 2: tops2}[file]['t_depart_sys'].isna().mean()

    @staticmethod
    def mmkn_blocking(lamda, mu, n_servers, capacity):
        """
        Computes the exact blocking probability of an M/M/K/N queue, that is the stationary probability that the
        system is full (an arriving agent sees it full, by PASTA). With n_servers = 1 it is an M/M/1/N queue.

        Args :
            lamda     : arrival rate.
            mu        : service rate of each server.
            n_servers : number of servers K.
            capacity  : total capacity N of the system, servers included (N >= K).

        Returns :
            The blocking probability, as a float.
        """
        weights = np.ones(capacity + 1)
        for n in range(1, capacity + 1):
            weights[n] = weights[n - 1] * lamda / (min(n, n_servers) * mu)
        return weights[-1] / weights.sum()

    @staticmethod
    def tandem_blocking_exact(lamda, mu1, n_servers, ks, mu2, kf):
        """
        Computes the exact refusal proportions of the tandem M/M/K/ks -> M/M/1/kf, from the stationary distribution
        of the Markov chain on (n1, n2), n1 being the number of pushes in file 1 and n2 the number of results in
        file 2. No Poisson assumption is made on the departures of file 1.

        Args :
            lamda, mu1, mu2 : arrival rate, service rate of each server of file 1, service rate of file 2.
            n_servers       : number K of servers of file 1.
            ks, kf          : total capacities of file 1 and file 2, servers included.

        Returns :
            (p1, p2) where p1 is the proportion of pushes refused by file 1 and p2 the proportion of results lost
            by file 2 among the results leaving file 1.
        """
        width = kf + 1
        size = (ks + 1) * width
        q = np.zeros((size, size))

        for n1 in range(ks + 1):
            for n2 in range(kf + 1):
                i = n1 * width + n2
                if n1 < ks:
                    q[i, (n1 + 1) * width + n2] += lamda
                if n1 > 0:
                    q[i, (n1 - 1) * width + min(n2 + 1, kf)] += min(n1, n_servers) * mu1
                if n2 > 0:
                    q[i, n1 * width + n2 - 1] += mu2
        q[np.diag_indices(size)] = -q.sum(axis=1)

        a = q.T.copy()
        a[-1, :] = 1.0
        b = np.zeros(size)
        b[-1] = 1.0
        pi = np.linalg.solve(a, b).reshape(ks + 1, width)

        departures = np.minimum(np.arange(ks + 1), n_servers) * mu1
        p1 = pi[ks].sum()
        p2 = (departures * pi[:, kf]).sum() / (departures * pi.sum(axis=1)).sum()
        return p1, p2
