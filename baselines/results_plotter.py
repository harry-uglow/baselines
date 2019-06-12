from __future__ import division
from __future__ import absolute_import
import numpy as np
import matplotlib
matplotlib.use(u'TkAgg') # Can change to 'Agg' for non-interactive mode

import matplotlib.pyplot as plt
plt.rcParams[u'svg.fonttype'] = u'none'

from baselines.common import plot_util

X_TIMESTEPS = u'timesteps'
X_EPISODES = u'episodes'
X_WALLTIME = u'walltime_hrs'
Y_REWARD = u'reward'
Y_TIMESTEPS = u'timesteps'
POSSIBLE_X_AXES = [X_TIMESTEPS, X_EPISODES, X_WALLTIME]
EPISODES_WINDOW = 100
COLORS = [u'blue', u'green', u'red', u'cyan', u'magenta', u'yellow', u'black', u'purple', u'pink',
        u'brown', u'orange', u'teal', u'coral', u'lightblue', u'lime', u'lavender', u'turquoise',
        u'darkgreen', u'tan', u'salmon', u'gold', u'darkred', u'darkblue']

def rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

def window_func(x, y, window, func):
    yw = rolling_window(y, window)
    yw_func = func(yw, axis=-1)
    return x[window-1:], yw_func

def ts2xy(ts, xaxis, yaxis):
    if xaxis == X_TIMESTEPS:
        x = np.cumsum(ts.l.values)
    elif xaxis == X_EPISODES:
        x = np.arange(len(ts))
    elif xaxis == X_WALLTIME:
        x = ts.t.values / 3600.
    else:
        raise NotImplementedError
    if yaxis == Y_REWARD:
        y = ts.r.values
    elif yaxis == Y_TIMESTEPS:
        y = ts.l.values
    else:
        raise NotImplementedError
    return x, y

def plot_curves(xy_list, xaxis, yaxis, title):
    fig = plt.figure(figsize=(8,2))
    maxx = max(xy[0][-1] for xy in xy_list)
    minx = 0
    for (i, (x, y)) in enumerate(xy_list):
        color = COLORS[i % len(COLORS)]
        plt.scatter(x, y, s=2)
        x, y_mean = window_func(x, y, EPISODES_WINDOW, np.mean) #So returns average of last EPISODE_WINDOW episodes
        plt.plot(x, y_mean, color=color)
    plt.xlim(minx, maxx)
    plt.title(title)
    plt.xlabel(xaxis)
    plt.ylabel(yaxis)
    plt.tight_layout()
    fig.canvas.mpl_connect(u'resize_event', lambda event: plt.tight_layout())
    plt.grid(True)


def split_by_task(taskpath):
    return taskpath[u'dirname'].split(u'/')[-1].split(u'-')[0]

def plot_results(dirs, num_timesteps=10e6, xaxis=X_TIMESTEPS, yaxis=Y_REWARD, title=u'', split_fn=split_by_task):
    results = plot_util.load_results(dirs)
    plot_util.plot_results(results, xy_fn=lambda r: ts2xy(r[u'monitor'], xaxis, yaxis), split_fn=split_fn, average_group=True, resample=int(1e6))

# Example usage in jupyter-notebook
# from baselines.results_plotter import plot_results
# %matplotlib inline
# plot_results("./log")
# Here ./log is a directory containing the monitor.csv files

def main():
    import argparse
    import os
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(u'--dirs', help=u'List of log directories', nargs = u'*', default=[u'./log'])
    parser.add_argument(u'--num_timesteps', type=int, default=int(10e6))
    parser.add_argument(u'--xaxis', help = u'Varible on X-axis', default = X_TIMESTEPS)
    parser.add_argument(u'--yaxis', help = u'Varible on Y-axis', default = Y_REWARD)
    parser.add_argument(u'--task_name', help = u'Title of plot', default = u'Breakout')
    args = parser.parse_args()
    args.dirs = [os.path.abspath(dir) for dir in args.dirs]
    plot_results(args.dirs, args.num_timesteps, args.xaxis, args.yaxis, args.task_name)
    plt.show()

if __name__ == u'__main__':
    main()
