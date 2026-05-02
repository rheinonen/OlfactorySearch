import pickle
import numpy as np

from matplotlib import pyplot as plt

def get_return(times,gamma=0.998,tmax=1000):
    t=np.array(times)
    rew=gamma**t-(1-gamma**t)
    rew[t>=tmax]=-1
    return rew

def get_fraction_success(times,tmax=1000):
    return np.sum(np.array(times)<tmax)/len(times)


def dist_to_source(x,y):
    x=np.asarray(x)
    y=np.asarray(y)
    return np.abs(y[:,0] - x[0]) + np.abs(y[:,1] - x[1]) - 2

def l1_to_discrete_disk(x, y, r):
    """
    Vectorized minimum L1 (Manhattan) distance from lattice point(s) x
    to the set of lattice points within Euclidean distance r of y.

    Computes
        min_{z in Z^2, ||z-y||_2 <= r} ||x-z||_1

    Parameters
    ----------
    x, y : array-like
        Points with final dimension 2. They may be broadcastable against each
        other. Examples:
            x.shape == (2,)
            y.shape == (N, 2)
        or
            x.shape == (..., 2)
            y.shape == (..., 2)
    r : float
        Disk radius.

    Returns
    -------
    out : ndarray or scalar
        Minimum number of lattice steps. Shape is the broadcasted shape of
        x[..., 0] and y[..., 0].
    """
    x = np.asarray(x)
    y = np.asarray(y)

    if x.shape[-1] != 2 or y.shape[-1] != 2:
        raise ValueError("x and y must have final dimension 2")

    if r < 0:
        raise ValueError("r must be nonnegative")

    # Broadcast x and y over all leading dimensions
    dx = x[..., 0] - y[..., 0]
    dy = x[..., 1] - y[..., 1]

    amax = int(np.floor(r))
    a = np.arange(-amax, amax + 1)  # candidate horizontal offsets in disk

    # For each a, allowable b satisfy |b| <= floor(sqrt(r^2 - a^2))
    bmax = np.floor(np.sqrt(r * r - a * a)).astype(int)

    # We minimize |dx - a| + min_{|b| <= bmax} |dy - b|
    # For fixed dy and interval [-bmax, bmax], the second term is
    # distance from dy to that interval: max(0, |dy| - bmax)
    #
    # Add a new trailing axis to dx,dy so they broadcast against a,bmax.
    cand = np.abs(dx[..., None] - a) + np.maximum(0, np.abs(dy[..., None]) - bmax)

    out = cand.min(axis=-1)
    return out.item() if out.ndim == 0 else out


def get_normalized_arrival(times,sources,ag_start=np.array([122,13]),tmax=1000):
    t=np.array(times)
    #distances = l1_to_discrete_disk(ag_start,sources,2)
    distances = dist_to_source(ag_start,sources)
    distances[t>=tmax] = 0
    return distances/t


tmax=500

cs_returns=[]
cs_successes=[]
cs_norm_times=[]

for thresh in [1,3,5,7]:
    cs_ret=[]
    cs_norm=[]
    cs_succ=[]
    with open('../results/results_marco_gamma_0.98_threshold_'+str(thresh)+'e-6_time_4000.pkl','rb') as f:
        data=pickle.load(f)
    print(f'threshold: {thresh}')
    print(f'quasi-optimal:')
    times=data['times']
    sources=data['sources']
    returns = get_return(times,tmax=tmax)
    norm_times = get_normalized_arrival(times,sources,tmax=tmax)
    frac_success = get_fraction_success(times,tmax=tmax)
    print(f'mean return: {np.mean(returns)}+/-{np.std(returns)/np.sqrt(len(returns))}')
    print(f'mean normalized time: {np.mean(norm_times)}+/-{np.std(norm_times)/np.sqrt(len(returns))}')
    print(f'fraction success: {frac_success}+/-{np.sqrt(frac_success*(1-frac_success)/len(times))}')
    print('cast-and-surge:')
    best_return=-1
    best_theta=None
    for theta in ['10','15','20','25','30','35','40','45','50','60','75','90']:
        with open('../results/results_marco_cs_theta_'+theta+'_threshold_'+str(thresh)+'e-6.pkl','rb') as f:
            data=pickle.load(f)
        times=data['times']
        sources=data['sources']
        returns = get_return(times,tmax=tmax)
        if np.mean(returns) > np.mean(best_return):
            best_theta = theta
            best_return= returns
            norm_times = get_normalized_arrival(times,sources,tmax=tmax)
            frac_success = get_fraction_success(times,tmax=tmax)
        cs_ret.append(np.mean(returns))
        cs_succ.append(frac_success)
        cs_norm.append(np.mean(norm_times))

    print(f'best theta: {best_theta}')
    print(f'mean return: {np.mean(best_return)}+/-{np.std(best_return)/np.sqrt(len(best_return))}')
    print(f'mean normalized time: {np.mean(norm_times)}+/-{np.std(norm_times)/np.sqrt(len(best_return))}')
    print(f'fraction success: {frac_success}+/-{np.sqrt(frac_success*(1-frac_success)/len(times))}')
    cs_returns.append(cs_ret)
    cs_successes.append(cs_succ)
    cs_norm_times.append(cs_norm)

out = {'returns':cs_results,'thetas':['10','15','20','25','30','35','40','45','50','60','75','90']}
with open('../results/cs_results.pkl','wb') as f:
    pickle.dump(out,f)

    

