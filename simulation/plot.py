#!/usr/bin/env python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from argparse import ArgumentParser

def parse_args():
    ap = ArgumentParser()
    ap.add_argument('history_file', nargs='+', help='CSV file with the simulation history')
    ap.add_argument('--plot-traj-3d', action='store_true', help='Enable 3D trajectory plot')
    ap.add_argument('--plot-traj-2d', action='store_true', help='Enable 2D trajectory plot')
    ap.add_argument('--plot-norm', action='store_true', help='Enable 2D norm plot')
    ap.add_argument('--errors', action='store_true', help='Values are deviations from the equilibrium')
    ap.add_argument('--controls', action='store_true', help='Values are active control values')
    return ap.parse_args()

def parse_quad(quad_str):
    return np.array([[float(x) for x in row.split(',')] for row in quad_str.split(';')])

def apply_quad(quad, x):
    x_colvec = np.array(x).reshape((-1, 1))
    return x_colvec.T() * quad * x_colvec

def plot_traj_2d(ax, t, x1, x2, x3, x):
    [line_x1] = ax.plot(t, x1, color='red')
    [line_x2] = ax.plot(t, x2, color='green')
    [line_x3] = ax.plot(t, x3, color='blue')
    plt.legend(
        handles=[line_x1, line_x2, line_x3],
        labels=[f'${x}_1(t)$', f'${x}_2(t)$', f'${x}_3(t)$'],
    )
    ax.set_xlabel('$t$')
    ax.set_ylabel(f'${x}_i(t)$')

def main():
    args = parse_args()
    histories = [pd.read_csv(history_file) for history_file in args.history_file]
    fig = plt.figure()

    if args.errors and args.controls:
        raise Exception('--errors and --controls are mutually exclusive')

    x = 'x'
    y = 'x'
    if args.errors:
        x = 'e'
    if args.controls:
        x = 'u'
        y = 'u'

    num_rows = sum(
        1 if flag else 0 for flag in [
            args.plot_traj_3d,
            args.plot_traj_2d,
            args.plot_norm,
        ]
    )
    index = 0

    if args.plot_traj_3d:
        index += 1
        ax = fig.add_subplot(num_rows, 1, index, projection='3d')
        for j, history in enumerate(histories):
            ax.plot(history[f'{y}1'], history[f'{y}2'], history[f'{y}3'])
        ax.set_xlabel(f'${x}_1$')
        ax.set_ylabel(f'${x}_2$')
        ax.set_zlabel(f'${x}_3$')

    if args.plot_traj_2d:
        index += 1
        ax = fig.add_subplot(num_rows, 1, index)
        for j, history in enumerate(histories):
            plot_traj_2d(ax, history['t'], history[f'{y}1'], history[f'{y}2'], history[f'{y}3'], x)

    if args.plot_norm:
        index += 1
        ax = fig.add_subplot(num_rows, 1, index)
        colors = ['teal', 'orange']
        lines = []
        labels = []
        for j, history in enumerate(histories):
            t = history['t']
            x1 = history[f'{y}1']
            x2 = history[f'{y}2']
            x3 = history[f'{y}3']
            [line] = ax.plot(t, (x1**2 + x2**2 + x3**2)**0.5, color=colors[j])
            lines.append(line)
            labels.append(f'Model {j+1}')
        ax.legend(handles=lines, labels=labels)
        ax.set_xlabel('$t$')
        ax.set_ylabel(f'$\\Vert {x}(t)\\Vert$')

    plt.show()

if __name__ == '__main__':
    main()
