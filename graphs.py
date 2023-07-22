import os
import glob

import numpy as np
import pandas as pd
import wntr
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.ticker import FormatStrFormatter
import viswaternet as vis

import utils
from metrics import Evaluator

pd.set_option("display.precision", 3)
COLORS = ['#94D1FE', '#3288c7', '#1d72af', '#12476d', '#0a1c29']
COLORS = ['#032962', '#005d8f', '#00aacc', '#70d8eb', '#b3e7f1']
COLORS = ["#032962", "#0175b3", "#01a9cb", "#5ed3e8", "#D7F6FE"]


def load_networks():
    networks = {}
    for file_path in glob.glob(os.path.join(SOLUTION_PATH, '*.inp')):
        base = os.path.basename(file_path)
        file_name, extension = os.path.splitext(base)
        net = wntr.network.WaterNetworkModel(file_path)
        networks[file_name] = net
    return networks


def get_leaks_repaired_in_year(networks, year):
    net_y = networks[f'y{str(year)}']
    net_prev_y = networks[f'y{str(year - 1)}']
    leaks_y = net_y.query_node_attribute('emitter_coefficient').dropna()
    leaks_prev_y = net_prev_y.query_node_attribute('emitter_coefficient').dropna()
    df = pd.merge(leaks_y.rename(f'y{year}'), leaks_prev_y.rename(f'y{year - 1}'),
                  left_index=True, right_index=True, how='right', indicator=True)

    return df.loc[df['_merge'] == 'right_only']


def get_leaks_summary(export_path=''):
    # get summary from raw network
    net = wntr.network.WaterNetworkModel(os.path.join('resources', 'networks', 'Base', 'BIWS_y0_base.inp'))
    evaluator = Evaluator([net])
    evaluator.run_hyd()
    leaks_summary = evaluator.leaks_summary()
    if export_path:
        leaks_summary.to_csv(export_path)
    return leaks_summary


def plot_leaks(networks, leaks_summary_path=''):
    if leaks_summary_path:
        df = pd.read_csv(leaks_summary_path)
    else:
        df = get_leaks_summary(networks['y0'])

    df.set_index('Leak_id', inplace=True)
    df['total_water_loss'] *= 3.6
    fig, axes = plt.subplots(nrows=2, ncols=3, sharex=True, sharey=True)
    axes = axes.ravel()
    for i, (file_name, net) in enumerate(networks.items()):
        axes[i].scatter(df['total_cost'], df['total_water_loss'], alpha=0.25, s=15, color='grey')
        if i > 0:
            repaired = get_leaks_repaired_in_year(networks, i)
            repaired = pd.merge(df, repaired, left_index=True, right_index=True, how='inner')
            axes[i].scatter(repaired['total_cost'], repaired['total_water_loss'], alpha=0.5, s=15,
                            color='dodgerblue', edgecolor='k', linewidth=0.5)

        axes[i].set_axisbelow(True)
        axes[i].grid()

    fig.tight_layout()

    # colors = ["#9CF1FA", '#0C75C0', "#0000FF", "#12476d", "#000080"]
    fig, ax = plt.subplots()
    sub_axes = plt.axes([.4, .59, .32, .26])
    _xmin, _xmax = 2250, 2520
    _ymin, _ymax = 0, 200

    df2 = df.loc[(df['total_cost'] > _xmin) & (df['total_cost'] < _xmax)]  # for zoomin plot
    df2 = df2.loc[(df2['total_water_loss'] > _ymin) & (df2['total_water_loss'] < _ymax)]  # for zoomin plot

    ax.scatter(df['total_cost'], df['total_water_loss'], s=29, color='none', edgecolor='red', linewidth=0.5)
    sub_axes.scatter(df2['total_cost'], df2['total_water_loss'], s=29, color='none', edgecolor='red', linewidth=0.5)
    for i, (file_name, net) in enumerate(networks.items()):
        if i > 0:
            repaired = get_leaks_repaired_in_year(networks, i)
            repaired = pd.merge(df, repaired, left_index=True, right_index=True, how='inner')
            ax.scatter(repaired['total_cost'], repaired['total_water_loss'], s=35,
                       color=COLORS[i - 1], edgecolor='k', linewidth=0.5)

            repaired2 = repaired.loc[(repaired['total_cost'] > _xmin) & (repaired['total_cost'] < _xmax)]
            repaired2 = repaired2.loc[(repaired2['total_water_loss'] > _ymin) & (repaired2['total_water_loss'] < _ymax)]
            sub_axes.scatter(repaired2['total_cost'], repaired2['total_water_loss'], s=35, color=COLORS[i - 1],
                             edgecolor='k', linewidth=0.5)

    plt.setp(sub_axes)
    ax.set_xlabel('Repair cost (€)')
    ax.set_ylabel('Total water loss at year 0 ($m^{3})$')

    cmap = mpl.colors.ListedColormap(COLORS)
    bounds = [1, 2, 3, 4, 5, 6]
    norm = mpl.colors.BoundaryNorm(bounds, 6)
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label('Repair year')
    cbar.ax.set_yticks([1.5, 2.5, 3.5, 4.5, 5.5])
    cbar.ax.set_yticklabels([1, 2, 3, 4, 5])
    # cbar.ax.set_yticklabels(bounds)


def metrics_by_year(score_file):
    titles = ['Effective hours proportion',
              'Continuous service pressure',
              'Leakage volume ($m^3)$',
              'Supply / Demand',
              'Consumers pressure level',
              'Continuous supply pressure',
              'Negative pressure pipes (km)',
              'Energy consumption (Mwh)',
              'Supply equity'
              ]

    df = pd.read_csv(score_file, index_col=0)
    fig, axes = plt.subplots(nrows=3, ncols=3, sharex=True, figsize=(9, 6))
    axes = axes.ravel()
    for i in range(9):
        axes[i].plot(df.iloc[:-1, i], marker='o', mfc='white')
        axes[i].set_title(f'I$_{{{i + 1}}}$-' + titles[i], fontsize=11)
        # axes[i].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        axes[i].grid()

    fig.text(0.5, 0.04, 'Year', ha='center')
    plt.subplots_adjust(left=0.1, right=0.95, wspace=0.25, hspace=0.35)


def plot_spatial_leaks():
    all_leaks = pd.read_csv('output/2_fcv/1_greedy_output/20230606080239_y1/leaks.csv', index_col=0)
    fig, axes = plt.subplots(nrows=1, ncols=2)

    temp_net = wntr.network.WaterNetworkModel(os.path.join(SOLUTION_PATH, 'y0.inp'))
    for leak in all_leaks.index:
        l = temp_net.get_node(leak)
        l.rank = all_leaks.loc[leak, 'rank']
    axes[0] = wntr.graphics.plot_network(temp_net, 'rank', ax=axes[0], node_range=[0.03, 0.1])

    temp_net = wntr.network.WaterNetworkModel(os.path.join(SOLUTION_PATH, 'y0.inp'))
    for year in [1, 2, 3, 4, 5]:
        repaired_leaks = get_leaks_repaired_in_year(all_networks, year)
        for leak in repaired_leaks.index:
            l = temp_net.get_node(leak)
            l.year = year
    axes[1] = wntr.graphics.plot_network(temp_net, 'year', ax=axes[1])


def plot_spatial_pipes():
    base_net = wntr.network.WaterNetworkModel(os.path.join('resources', 'networks', 'BIWS.inp'))
    all_actions = pd.read_csv('output/2_fcv/all_actions.csv')
    replaced = all_actions.loc[all_actions['action'] == 'pipe']
    replaced = replaced.drop_duplicates(subset='element')

    all_pipes = base_net.query_link_attribute(attribute='length', link_type=wntr.network.Pipe).rename('length')
    all_pipes = pd.merge(all_pipes, replaced[['element', 'year']], left_index=True, right_on='element', how='outer')
    all_pipes.set_index('element', inplace=True)
    all_pipes['year'].fillna(0, inplace=True)

    all_pipes.reset_index(inplace=True)

    wntr.network.write_inpfile(base_net, 'test.inp')

    # wntr.graphics.plot_network(base_net, link_attribute='year', node_size=8, link_width=10)

    fig, ax = plt.subplots()
    colors = ["k", "#0be7d9", "#067eb1", "#369661", "#f8961e", "#f81215"]
    model = vis.VisWNModel(os.path.join('resources', 'networks', 'BIWS.inp'))
    model.plot_unique_data(ax=ax, parameter='custom_data', node_size=20,
                           parameter_type='link', data_type='discrete',
                           edge_colors="k",
                           line_widths=0,
                           custom_data_values=[all_pipes['element'], all_pipes['year']],
                           color_list=colors,
                           node_shape='o',
                           valves=False,
                           reservoirs=True,
                           reservoir_size=40,
                           reservoir_border_width=0.5,
                           reservoir_color='dodgerblue',
                           pumps=False,
                           tank_size=50,
                           tank_border_width=0.5,
                           interval_link_width_list=[1, 2, 2, 2, 2, 2],
                           tank_shape='o',
                           tank_color='dodgerblue',
                           legend=False,
                           draw_frame=False
                           )

    cmap = mpl.colors.ListedColormap(colors[1:])
    bounds = [1, 2, 3, 4, 5, 6]
    norm = mpl.colors.BoundaryNorm(bounds, 6)
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.05)
    cbar.set_label('Repair year')
    # cbar.ax.set_yticks([1.5, 2.5, 3.5, 4.5, 5.5])
    # cbar.ax.set_yticklabels([1, 2, 3, 4, 5])


def plot_iterations(iterations_file):
    df = pd.read_csv(iterations_file)
    df = df.rename(columns={col: int(col) for col in [str(_) for _ in range(1, 10)]})
    obj_cols = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    worse = {1: 0, 2: 0, 3: 1, 4: 0, 5: 0, 6: 0, 7: 161630.3, 8: 16688.6, 9: 0}
    for i, row in df.iterrows():
        if i == 0:
            df.loc[i, 'so'] = 9
        else:
            df.loc[i, 'so'] = sum(utils.normalize_obj(row[obj_cols].to_dict(),
                                                      df.loc[i - 1, obj_cols].to_dict()).values())
            df.loc[i, 'so_'] = sum(utils.normalize_obj(row[obj_cols].to_dict(),
                                                       worse).values())

    df['db'] = 9 - df['so']
    df['db_'] = df['so_'] - df['so'].shift()
    fig, ax = plt.subplots()
    temp = df.loc[df['year'] == 1]
    min_idx, max_idx = temp.index.min(), temp.index.max()
    plt.axvspan(min_idx, max_idx, color='grey', alpha=0.4, lw=0)

    temp = df.loc[df['year'] == 3]
    min_idx, max_idx = temp.index.min(), temp.index.max()
    plt.axvspan(min_idx, max_idx, color='grey', alpha=0.4, lw=0)

    temp = df.loc[df['year'] == 5]
    min_idx, max_idx = temp.index.min(), temp.index.max()
    plt.axvspan(min_idx, max_idx, color='grey', alpha=0.4, lw=0)

    ax.plot(df['db_'])
    # ax.plot(df['evaluations'])


def plot_evaluations_per_pipe():
    df = pd.DataFrame(index=all_pipes['ID'])
    for y, greedy_results in enumerate(all_greedy):
        year_flags = pd.DataFrame(index=all_pipes['ID'])
        for filename in os.listdir(greedy_results):
            if filename.startswith("pipes_flags"):
                i = filename.split('_')[1]
                flags = pd.read_csv(os.path.join(greedy_results, filename), index_col=0)
                year_flags = pd.concat([year_flags, flags['evaluate_flag'].rename(str(i))], axis=1)
            else:
                continue

        year_flags = year_flags.sum(axis=1)
        df = pd.concat([df, year_flags.rename(f'Year {str(y + 1)}')], axis=1)

    df['s'] = df.sum(axis=1)
    df = df.sort_values('s', ascending=False)
    df = df.loc[df['s'] >= 20]
    df = df.drop('s', axis=1)

    df.plot(kind='bar', stacked=True, color=COLORS, edgecolor='k', width=0.6, figsize=(8, 4), linewidth=0.6)
    plt.subplots_adjust(bottom=0.2, left=0.08, right=0.98)
    plt.ylabel('Evaluations Count')


if __name__ == "__main__":
    y1_greedy = os.path.join('output', '2_fcv', '1_greedy_output', '20230606080239_y1')
    y2_greedy = os.path.join('output', '2_fcv', '1_greedy_output', '20230608092158_y2')
    y3_greedy = os.path.join('output', '2_fcv', '1_greedy_output', '20230610160100_y3')
    y4_greedy = os.path.join('output', '2_fcv', '1_greedy_output', '20230611104545_y4')
    y5_greedy = os.path.join('output', '2_fcv', '1_greedy_output', '20230613112601_y5')
    all_greedy = [y1_greedy, y2_greedy, y3_greedy, y4_greedy, y5_greedy]

    SOLUTION_PATH = os.path.join('output/2_fcv/2_final_networks')
    all_networks = load_networks()
    all_pipes = pd.read_csv('resources/pipes_data.csv')

    # get_leaks_repaired_in_year(all_networks, 1)
    # get_leaks_summary(export_path='output/fcv/data_for_figures/leaks_summary.csv')
    # plot_leaks(all_networks, leaks_summary_path='resources/all_leaks_summary.csv')

    # metrics_by_year('output/2_fcv/score.csv')
    # plot_iterations('output/2_fcv/all_iter.csv')
    # plot_spatial_pipes()
    # plot_spatial_leaks()

    # plot_evaluations_per_pipe()  # in paper

    # all_actions = pd.read_csv('output/2_fcv/all_actions.csv')
    # all_leaks = pd.read_csv('resources/all_leaks_summary.csv')

    temp_net = wntr.network.WaterNetworkModel(os.path.join('resources', 'networks', 'BIWS.inp'))
    df = pd.read_csv(os.path.join('output', '2_fcv', 'pipes_repair_year.csv'), index_col=0)
    all_pipes = temp_net.query_link_attribute(attribute='length', link_type=wntr.network.Pipe).rename('length')
    all_pipes = pd.merge(all_pipes, df, left_index=True, right_on='pipe', how='outer')
    all_pipes = all_pipes.fillna(0)
    all_pipes.set_index('pipe', inplace=True)
    all_pipes = all_pipes.drop('length', axis=1)
    print(all_pipes)
    all_pipes.to_csv(os.path.join('output', '2_fcv', 'pipes_repair_year.csv'))
    # for pipe_id in df.index:
    #     pipe = temp_net.get_link(pipe_id)
    #     pipe.year = df.loc[pipe_id]

    # for year in [1, 2, 3, 4, 5]:
    #     repaired_leaks = get_leaks_repaired_in_year(all_networks, year)
    #     for leak in repaired_leaks.index:
    #         l = temp_net.get_node(leak)
    #         l.year = year

    # wntr.network.io.write_inpfile(temp_net, os.path.join('output', '2_fcv', 'pipes_repair_year.inp'))
    plt.show()
