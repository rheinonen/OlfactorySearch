import numpy as np

from scipy.special import softmax

import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde


import numpy as np
import matplotlib.pyplot as plt


import matplotlib as mpl
mpl.rcParams['ytick.labelsize'] = 20 

def reflected_weighted_kde(x, w=None, bw_method=None):
    x = np.asarray(x, dtype=float)
    if w is None:
        w = np.ones_like(x)
    else:
        w = np.asarray(w, dtype=float)

    mask = np.isfinite(x) & np.isfinite(w)
    x = x[mask]
    w = w[mask]

    if np.any(x < 0):
        raise ValueError("Reflection method here assumes support [0, ∞).")
    if np.any(w < 0):
        raise ValueError("weights must be nonnegative")
    if np.sum(w) == 0:
        raise ValueError("sum of weights must be positive")

    w = w / np.sum(w)

    # augmented reflected sample
    x_aug = np.concatenate([x, -x])
    w_aug = np.concatenate([0.5 * w, 0.5 * w])

    kde = gaussian_kde(x_aug, weights=w_aug, bw_method=bw_method)
    return kde

def weighted_quantile(x, w, q):
    """
    Weighted quantile of x at probability q in [0, 1].

    Uses the left-continuous weighted empirical CDF.
    """
    x = np.asarray(x, dtype=float)
    w = np.asarray(w, dtype=float)

    if x.ndim != 1 or w.ndim != 1:
        raise ValueError("x and w must be 1D")
    if len(x) != len(w):
        raise ValueError("x and w must have the same length")
    if np.any(w < 0):
        raise ValueError("weights must be nonnegative")

    mask = np.isfinite(x) & np.isfinite(w)
    x = x[mask]
    w = w[mask]

    if len(x) == 0:
        raise ValueError("no valid data points")
    if np.sum(w) == 0:
        raise ValueError("sum of weights must be positive")

    order = np.argsort(x)
    x = x[order]
    w = w[order]

    cw = np.cumsum(w)
    cw = cw / cw[-1]

    idx = np.searchsorted(cw, q, side="left")
    idx = min(idx, len(x) - 1)
    return x[idx]


def weighted_boxplot_stats(x, w=None, whis=1.5, label=None, showfliers=True):
    """
    Compute weighted boxplot stats suitable for matplotlib.axes.Axes.bxp.
    """
    x = np.asarray(x, dtype=float)

    if w is None:
        w = np.ones_like(x, dtype=float)
    else:
        w = np.asarray(w, dtype=float)

    if len(x) != len(w):
        raise ValueError("x and w must have the same length")
    if np.any(w < 0):
        raise ValueError("weights must be nonnegative")

    mask = np.isfinite(x) & np.isfinite(w)
    x = x[mask]
    w = w[mask]

    if len(x) == 0:
        raise ValueError("no valid data points")
    if np.sum(w) == 0:
        raise ValueError("sum of weights must be positive")

    q1 = weighted_quantile(x, w, 0.25)
    med = weighted_quantile(x, w, 0.50)
    q3 = weighted_quantile(x, w, 0.75)
    mean = np.sum(w * x) / np.sum(w)

    iqr = q3 - q1
    lo_fence = q1 - whis * iqr
    hi_fence = q3 + whis * iqr

    inside = x[(x >= lo_fence) & (x <= hi_fence)]

    if len(inside) == 0:
        whislo = q1
        whishi = q3
    else:
        whislo = np.min(inside)
        whishi = np.max(inside)

    fliers = x[(x < whislo) | (x > whishi)] if showfliers else np.array([])

    return {
        "label": label,
        "mean": mean,
        "med": med,
        "q1": q1,
        "q3": q3,
        "whislo": whislo,
        "whishi": whishi,
        "fliers": fliers,
    }


def weighted_boxplot(
    ax,
    datasets,
    weights=None,
    labels=None,
    whis=1.5,
    positions=None,
    widths=0.5,
    showmeans=False,
    showfliers=True,
    patch_artist=True,
    **kwargs,
):
    """
    Draw weighted boxplots using matplotlib's bxp from precomputed stats.
    """
    datasets = [np.asarray(d, dtype=float) for d in datasets]
    n = len(datasets)

    if weights is None:
        weights = [None] * n
    elif len(weights) != n:
        raise ValueError("weights must have same length as datasets")

    if labels is None:
        labels = [None] * n
    elif len(labels) != n:
        raise ValueError("labels must have same length as datasets")

    stats = [
        weighted_boxplot_stats(x, w=w, whis=whis, label=lab, showfliers=showfliers)
        for x, w, lab in zip(datasets, weights, labels)
    ]

    artists = ax.bxp(
        stats,
        positions=positions,
        widths=widths,
        showmeans=showmeans,
        showfliers=showfliers,
        patch_artist=patch_artist,
        **kwargs,
    )

    return artists

def weighted_violinplot(
    ax,
    datasets,
    weights=None,
    positions=None,
    widths=0.5,
    points=200,
    bw_method=None,
    showmedians=False,
    showmeans=False,
    showextrema=False,
    vert=True,
    **fill_kwargs,
):
    datasets = [np.asarray(d, dtype=float) for d in datasets]
    n = len(datasets)

    if weights is None:
        weights = [np.ones_like(d) for d in datasets]
    else:
        weights = [np.asarray(w, dtype=float) for w in weights]

    if positions is None:
        positions = np.arange(1, n + 1)

    artists = []
    mean_lines = []
    median_lines = []
    min_lines = []
    max_lines = []
    bar_lines = []

    default_fill = dict(alpha=0.4, linewidth=1)
    default_fill.update(fill_kwargs)

    for pos, x, w in zip(positions, datasets, weights):
        mask = np.isfinite(x) & np.isfinite(w)
        x = x[mask]
        w = w[mask]

        if len(x) == 0:
            continue
        if len(x) != len(w):
            raise ValueError("each weights array must match its dataset")
        if np.any(w < 0):
            raise ValueError("weights must be nonnegative")
        if np.sum(w) == 0:
            raise ValueError("sum of weights must be positive")

        w = w / np.sum(w)

        if np.allclose(x, x[0]):
            grid = np.array([x[0] - 1e-8, x[0], x[0] + 1e-8])
            dens = np.array([0.0, 1.0, 0.0])
        else:
             kde = reflected_weighted_kde(x, w=w, bw_method=bw_method)
             grid = np.linspace(0.0, np.max(x), points)
             dens = kde(grid)
            #kde = gaussian_kde(x, weights=w, bw_method=bw_method)
            #xmin, xmax = np.min(x), np.max(x)
            #pad = 0.1 * (xmax - xmin) if xmax > xmin else 1.0
            #grid = np.linspace(xmin - pad, xmax + pad, points)
            #dens = kde(grid)

        dens = dens / dens.max() * widths/2

        if vert:
            body = ax.fill_betweenx(grid, pos - dens, pos + dens, **default_fill)
        else:
            body = ax.fill_between(grid, pos - dens, pos + dens, **default_fill)
        artists.append(body)

        if showmeans:
            mean = np.sum(w * x)
            if vert:
                line, = ax.plot([pos - widths/4, pos + widths/4], [mean, mean], lw=1.5)
            else:
                line, = ax.plot([mean, mean], [pos - widths/4, pos + widths/4], lw=1.5)
            mean_lines.append(line)

        if showmedians:
            idx = np.argsort(x)
            xs = x[idx]
            ws = w[idx]
            cdf = np.cumsum(ws)
            median = xs[np.searchsorted(cdf, 0.5)]
            if vert:
                line, = ax.plot([pos - widths/2, pos + widths/2], [median, median], lw=1)
            else:
                line, = ax.plot([median, median], [pos - widths/2, pos + widths/2], lw=1)
            median_lines.append(line)

        if showextrema:
            xmin_data, xmax_data = np.min(x), np.max(x)
            if vert:
                bar, = ax.plot([pos, pos], [xmin_data, xmax_data], lw=1)
                mn, = ax.plot([pos - widths/2, pos + widths/2], [xmin_data, xmin_data], lw=1)
                mx, = ax.plot([pos - widths/2, pos + widths/2], [xmax_data, xmax_data], lw=1)
            else:
                bar, = ax.plot([xmin_data, xmax_data], [pos, pos], lw=1)
                mn, = ax.plot([xmin_data, xmin_data], [pos - widths/2, pos + widths/2], lw=1)
                mx, = ax.plot([xmax_data, xmax_data], [pos - widths/2, pos + widths/2], lw=1)
            bar_lines.append(bar)
            min_lines.append(mn)
            max_lines.append(mx)

    if vert:
        ax.set_xticks(positions)
    else:
        ax.set_yticks(positions)

    return {
        'bodies': artists,
        'cmeans': mean_lines,
        'cmedians': median_lines,
        'cbars': bar_lines,
        'cmins': min_lines,
        'cmaxes': max_lines,
    }

def m(x, w):
    """Weighted Mean"""
    return np.sum(x * w) / np.sum(w)

def cov(x, y, w):
    """Weighted Covariance"""
    return np.sum(w * (x - m(x, w)) * (y - m(y, w))) / np.sum(w)

def corr(x, y, w):
    """Weighted Correlation"""
    return cov(x, y, w) / np.sqrt(cov(x, x, w) * cov(y, y, w))

def get_stats(fun,trajs,weights):
    lens=[]
    for traj in trajs:
        lens.append(fun(traj))
    lens=np.array(lens)
    weights_array=np.array(weights)
    weights_array[np.isnan(lens)|np.isnan(weights)]=0
    if np.sum(weights_array)==0:
        print('warning: zero weight sum')
        return[np.nan,np.nan,np.nan,np.isnan(lens)]
    means=bootstrap(np.mean,lens,weights=weights_array)
    #assert np.mean(means)>=np.quantile(means,0.025) and np.mean(means)<=np.quantile(means,0.975), f'bootstrap mean {np.mean(means)}, quantiles {np.quantile(means,0.025)} and {np.quantile(means,0.975)}'
    return [np.mean(means),max(np.mean(means)-np.quantile(means,0.025),0),max(np.quantile(means,0.975)-np.mean(means),0),np.isnan(lens)]


def bootstrap(stat,a,num=9999,weights=None):
    if weights is not None:
        assert not np.isnan(weights).any()
        assert not np.isnan(np.array(weights)**2).any(),np.array(weights)[np.isnan(np.array(weights))]
        size=int(np.floor(np.sum(weights)**2/np.sum(np.array(weights)**2)))
        stats=[stat(np.random.choice(a,size=size,p=np.array(weights)/np.sum(weights))) for _ in range(num)]
    else:
        stats=[stat(np.random.choice(a,size=len(a))) for _ in range(num)]
    return stats


def compute_surge_length(traj,num_accepted_steps=3):
    num_left=0
    num_updown=0
    last_pos=None
    for pos in traj:
        if last_pos is not None:
            action=pos-last_pos
            if (action==np.array([-1,0])).all():
                num_left+=1
            elif (action==np.array([1,0])).all():
                num_left-=1
            else:
                num_updown+=1
            if num_updown>=num_accepted_steps:
                break
        last_pos=pos
    return max(num_left,0)

def compute_backtracking_length(traj):
    pos = np.zeros(2)
    min_pos = (0, pos[0], pos.copy())

    for i in range(1, len(traj)):
        action = traj[i] - traj[i - 1]
        pos += action
        if pos[0] < min_pos[1]:
            min_pos = (i, pos[0], pos.copy())

    pos = min_pos[2].copy()
    back_pos = (min_pos[0], pos[0], pos.copy())

    for i in range(min_pos[0] + 1, len(traj)):
        action = traj[i] - traj[i - 1]
        pos += action
        if pos[0] > back_pos[1]:
            back_pos = (i, pos[0], pos.copy())

    return back_pos[1] - min_pos[1], back_pos[0]


def compute_backtracking_init_time(traj, acceptance_steps = 2,count_from=1):
    pos = np.zeros(2)
    num_right = 0
    for i in range(count_from, len(traj)):
        action=traj[i]-traj[i-1]
        if (action==np.array([1,0])).all():
            num_right += 1
        elif (action==np.array([-1,0])).all():
            num_right = 0 
        if num_right > acceptance_steps:
            break
    if i>=len(traj)-1:
        return np.nan

    return max(i - acceptance_steps - count_from, 0)

def compute_search_length(traj, count_from=1):
    pos = np.zeros(2)
    max_pos = 0
    for i in range(count_from, len(traj)):
        pos += traj[i]-traj[i-1]
        max_pos = min(pos[0], max_pos)
    return abs(max_pos)

def compute_cast_width(traj, count_from=1):
    max_pos = 0.0
    min_pos = 0.0
    pos = np.zeros(2)
    for i in range(count_from, len(traj)):
        pos += traj[i]-traj[i-1]
        max_pos = max(pos[1], max_pos, 0)
        min_pos = min(pos[1], min_pos, 0)
    return max_pos + abs(min_pos)

thresholds=['1e-6','3e-6','5e-6','7e-6']



import pickle
from matplotlib import pyplot as plt

fig,ax=plt.subplots(5,1,layout='constrained',figsize=(4,10))
horizon=100
gamma=0.98

median_colors = [
    '#00eff1',
    '#995555',
    'lightgreen',
    '#ffffff'
]

colors=plt.rcParams['axes.prop_cycle'].by_key()['color']
colors[3] = 'gray'
colors = [
    '#4D7F8D',
    '#D6833B',
    '#3E6F1D',
    '#515151'
    ]

from scipy.stats import pearsonr

all_surge_lens=[]
all_backtrack_lens=[]
all_backtrack_times=[]
all_search_lens=[]
all_cast_widths=[]
all_weights = []

for threshold,color in zip(thresholds,colors):
    print(threshold)
    surge_lens=[]
    backtrack_lens=[]
    backtrack_times=[]
    search_lens=[]
    cast_widths=[]
    x_hats=[]
    y_hats=[]
    sigma_xs=[]
    sigma_ys=[]
    weights=[]
    with open(f'../results/conditioned_mc_new_threshold_'+str(threshold)+'.pkl','rb') as f:
        data=pickle.load(f)
#       print(data['nohit_weights'],sum(data['nohit_weights']))
    for traj,x_hat,y_hat,sigma_x,sigma_y,weight,log_nohit_w in zip(data['trajs'],data['x_hat'],data['y_hat'],data['sigma_x'],data['sigma_y'],data['weights'],data['log_nohit_weights']):
        if traj==[]:
            continue
        x_hats.append(-x_hat)
        y_hats.append(np.abs(y_hat))
        sigma_xs.append(sigma_x)
        sigma_ys.append(sigma_y)
        if weight==0 or np.any(np.isnan(log_nohit_w)):
            continue
        bt=compute_backtracking_init_time(traj)
        if np.isnan(bt):
            continue
        else:
            surge_lens.append(compute_surge_length(traj))
            backtrack_times.append(bt)
            bl, t_traj = compute_backtracking_length(traj)
            assert bl >= 0, f'backtrack length is {bl}'
            assert isinstance(t_traj,int),f't_traj is {t_traj}'
            backtrack_lens.append(bl)
            weights.append(np.sum(log_nohit_w[:t_traj]))
            search_lens.append(compute_search_length(traj))
            cast_widths.append(compute_cast_width(traj))
    ys=[surge_lens,backtrack_lens,backtrack_times,search_lens,cast_widths]
    all_surge_lens.append(surge_lens)
    all_backtrack_lens.append(backtrack_lens)
    all_search_lens.append(search_lens)
    all_backtrack_times.append(backtrack_times)
    all_cast_widths.append(cast_widths)
    all_weights.append(softmax(weights))


plots=[]
#plots.append(weighted_boxplot(ax[0],all_surge_lens,weights=all_weights,showfliers=True,patch_artist=True,showmeans=False))
#plots.append(weighted_boxplot(ax[3],all_backtrack_lens,weights=all_weights,showfliers=True,showmeans=False,patch_artist=True))
#plots.append(weighted_boxplot(ax[4],all_backtrack_times,weights=all_weights,showfliers=True,showmeans=False,patch_artist=True))
#plots.append(weighted_boxplot(ax[2],all_search_lens,weights=all_weights,showfliers=True,showmeans=False,patch_artist=True))
#plots.append(weighted_boxplot(ax[1],all_cast_widths,weights=all_weights,showfliers=True,showmeans=False,patch_artist=True))

plots.append(weighted_violinplot(ax[0],all_surge_lens,weights=all_weights,showmeans=True,showextrema=False))
plots.append(weighted_violinplot(ax[3],all_backtrack_lens,weights=all_weights,showmeans=True,showextrema=False))
plots.append(weighted_violinplot(ax[4],all_backtrack_times,weights=all_weights,showmeans=True,showextrema=False))
plots.append(weighted_violinplot(ax[2],all_search_lens,weights=all_weights,showmeans=True,showextrema=False))
plots.append(weighted_violinplot(ax[1],all_cast_widths,weights=all_weights,showmeans=True,showextrema=False))

#ax[0].set_ylabel('Surge upwind')
#ax[3].set_ylabel('Downwind length')
#ax[4].set_ylabel('Downwind init time')
#ax[2].set_ylabel('Upwind search')
#ax[1].set_ylabel('Crosswind cast')

for i in range(5):
# Set the color of the violin patches
    for pc, color in zip(plots[i]['bodies'], colors):
        pc.set_facecolor(color)
        pc.set_alpha(0.9)

    # Set the color of the median lines
    for line, color in zip(plots[i]['cmeans'], median_colors):
        line.set_color(color)


for i in range(5):
    ax[i].set_xticks(ticks=[],labels=[])
ax[4].set_xticks([1,2,3,4],labels=[])
    #ax[i].set_xticklabels(['Denser','Dense','Sparse','Sparser'])
fig.savefig(f'../results/geometry_distribution_horizon_100_weighted_variably.pdf')
