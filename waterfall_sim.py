"""
Waterfall : réseau de files en tandem
  File 1 : arrivées Poisson(lam), K serveurs (test-suite), attente FIFO de capacité ks (None = infinie)
  [Back-up optionnel : avec proba p, le résultat passe par un back-up (nb serveurs, taux mub)]
  File 2 : 1 serveur (envoi vers le front), attente FIFO de capacité kf (None = infinie)

Refus en file 1  -> message d'erreur au push
Refus en file 2  -> page blanche, sauf si le résultat a été sauvegardé (back-up) : il est alors
                    récupéré via un chemin lent (délai exponentiel de taux mur, hors file 2).
Unités de temps : secondes.
"""
import heapq, math, random, statistics
from collections import deque

ARR, DEP1, DEPB, DEP2, REC = range(5)


def draw_service(rng, mean, kind="exp", cv=1.0):
    if kind == "exp":
        return rng.expovariate(1.0 / mean)
    if kind == "det":
        return mean
    if kind == "logn":  # lognormale de moyenne 'mean' et coefficient de variation cv
        s2 = math.log(1 + cv * cv)
        return rng.lognormvariate(math.log(mean) - s2 / 2, math.sqrt(s2))
    raise ValueError(kind)


def simulate(lam, mu1, K, ks, mu2, kf, p=0.0, mub=None, nb=1, mur=None,
             T=200_000.0, warm=20_000.0, seed=0, svc1="exp", cv1=1.0):
    rng = random.Random(seed)
    mur = mur or mu2
    ev = []
    cnt = 0

    def push(t, kind, cid):
        nonlocal cnt
        cnt += 1
        heapq.heappush(ev, (t, cnt, kind, cid))

    push(rng.expovariate(lam), ARR, 0)
    t_arr = {}
    backed = {}
    # file 1
    busy1 = 0
    wait1 = deque()
    # backup
    busyb = 0
    waitb = deque()
    # file 2
    busy2 = False
    wait2 = deque()

    cid_next = 1
    st = dict(arrivals=0, refused1=0, blank=0, recovered=0, delivered=0,
              backed=0, arrived2=0, refused2=0)
    sojourn, soj_rec, soj_direct = [], [], []
    wq1, wq2 = [], []
    enter_q1, enter_q2 = {}, {}
    # intégrale du nombre en file 2 / backup (taux d'occupation)
    last_t = 0.0
    area_b = 0.0
    nb_in_backup = 0

    def counted(cid):
        return t_arr[cid] >= warm

    def start1(cid, t):
        nonlocal busy1
        busy1 += 1
        if counted(cid):
            wq1.append(t - enter_q1[cid])
        push(t + draw_service(rng, 1.0 / mu1, svc1, cv1), DEP1, cid)

    def arrive2(cid, t):
        nonlocal busy2
        if counted(cid):
            st["arrived2"] += 1
        enter_q2[cid] = t
        if not busy2:
            busy2 = True
            if counted(cid):
                wq2.append(0.0)
            push(t + rng.expovariate(mu2), DEP2, cid)
        elif kf is None or len(wait2) < kf:
            wait2.append(cid)
        else:  # refus
            if counted(cid):
                st["refused2"] += 1
            if backed.get(cid):
                if counted(cid):
                    st["recovered"] += 1
                push(t + rng.expovariate(mur), REC, cid)
            else:
                if counted(cid):
                    st["blank"] += 1

    def startb(cid, t):
        nonlocal busyb
        busyb += 1
        push(t + rng.expovariate(mub), DEPB, cid)

    while ev:
        t, _, kind, cid = heapq.heappop(ev)
        if t > T:
            break
        if t >= warm:
            area_b += nb_in_backup * (t - max(last_t, warm))
        last_t = t
        if kind == ARR:
            i = cid_next
            cid_next += 1
            t_arr[i] = t
            push(t + rng.expovariate(lam), ARR, 0)
            if t >= warm:
                st["arrivals"] += 1
            if busy1 < K:
                enter_q1[i] = t
                start1(i, t)
            elif ks is None or len(wait1) < ks:
                enter_q1[i] = t
                wait1.append(i)
            else:
                if t >= warm:
                    st["refused1"] += 1
        elif kind == DEP1:
            busy1 -= 1
            if wait1:
                start1(wait1.popleft(), t)
            if p > 0 and rng.random() < p:
                backed[cid] = True
                if counted(cid):
                    st["backed"] += 1
                nb_in_backup += 1
                if busyb < nb:
                    startb(cid, t)
                else:
                    waitb.append(cid)
            else:
                arrive2(cid, t)
        elif kind == DEPB:
            busyb -= 1
            nb_in_backup -= 1
            if waitb:
                startb(waitb.popleft(), t)
            arrive2(cid, t)
        elif kind == DEP2:
            if counted(cid):
                s = t - t_arr[cid]
                sojourn.append(s)
                (soj_rec if False else soj_direct).append(s)
                st["delivered"] += 1
            if wait2:
                nxt = wait2.popleft()
                if counted(nxt):
                    wq2.append(t - enter_q2[nxt])
                push(t + rng.expovariate(mu2), DEP2, nxt)
            else:
                busy2 = False
        elif kind == REC:
            if counted(cid):
                s = t - t_arr[cid]
                sojourn.append(s)
                soj_rec.append(s)
    n = st["arrivals"]
    acc = n - st["refused1"]
    res = dict(
        P_refus_push=st["refused1"] / n if n else float("nan"),
        P_blank_acc=st["blank"] / acc if acc else float("nan"),     # parmi les pushs acceptés
        P_blank_all=st["blank"] / n if n else float("nan"),         # parmi tous les pushs
        P_refus2=st["refused2"] / st["arrived2"] if st["arrived2"] else float("nan"),
        P_recovered=st["recovered"] / acc if acc else float("nan"),
        W=statistics.fmean(sojourn) if sojourn else float("nan"),
        Var=statistics.pvariance(sojourn) if len(sojourn) > 1 else float("nan"),
        Wq1=statistics.fmean(wq1) if wq1 else float("nan"),
        Wq2=statistics.fmean(wq2) if wq2 else float("nan"),
        N_backup=area_b / (T - warm),
    )
    return res


def replicate(R=8, **kw):
    out = [simulate(seed=s, **kw) for s in range(R)]
    agg = {}
    for k in out[0]:
        v = [o[k] for o in out if not math.isnan(o[k])]
        if not v:
            agg[k] = (float("nan"), float("nan"))
            continue
        m = statistics.fmean(v)
        h = 1.96 * statistics.stdev(v) / math.sqrt(len(v)) if len(v) > 1 else 0.0
        agg[k] = (m, h)
    return agg


# ----------------------------------------------------------------- théorie
def erlang_c_wq(lam, mu, K):
    a = lam / mu
    rho = a / K
    if rho >= 1:
        return float("inf")
    s = sum(a ** n / math.factorial(n) for n in range(K))
    top = a ** K / (math.factorial(K) * (1 - rho))
    pw = top / (s + top)
    return pw / (K * mu - lam)


def mmkN_block(lam, mu, K, N):
    a = lam / mu
    w = []
    for n in range(N + 1):
        if n <= K:
            w.append(a ** n / math.factorial(n))
        else:
            w.append(a ** K / math.factorial(K) * (a / K) ** (n - K))
    return w[-1] / sum(w)


def mm1M_block(lam, mu, M):
    rho = lam / mu
    if abs(rho - 1) < 1e-12:
        return 1 / (M + 1)
    return (1 - rho) * rho ** M / (1 - rho ** (M + 1))
