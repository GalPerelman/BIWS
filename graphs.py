import os
import glob

import numpy as np
import pandas as pd
import wntr
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import FormatStrFormatter

import viswaternet as vis

import utils
from metrics import Evaluator

pd.set_option("display.precision", 3)
COLORS = ['#94D1FE', '#3288c7', '#1d72af', '#12476d', '#0a1c29']
COLORS = ['#032962', '#005d8f', '#00aacc', '#70d8eb', '#b3e7f1']
COLORS = ["#032962", "#0175b3", "#01a9cb", "#5ed3e8", "#D7F6FE"][::-1]


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
    sub_axes = plt.axes([.36, .57, .36, .28])
    _xmin, _xmax = 2100, 2810
    _ymin, _ymax = 0, 150

    df2 = df.loc[(df['total_cost'] > _xmin) & (df['total_cost'] < _xmax)]  # for zoomin plot
    df2 = df2.loc[(df2['total_water_loss'] > _ymin) & (df2['total_water_loss'] < _ymax)]  # for zoomin plot

    ax.scatter(df['total_cost'], df['total_water_loss'], s=28, color='none', edgecolor='red', linewidth=0.5)
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

    fig.text(0.5, 0.03, 'Year', ha='center')
    plt.subplots_adjust(left=0.1, right=0.95, top=0.95, wspace=0.25, hspace=0.35)


def plot_spatial_leaks():
    all_leaks = pd.read_csv('output/1_greedy_output/20230606080239_y1/leaks.csv', index_col=0)
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
    all_actions = pd.read_csv('output/all_actions.csv')
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

    fig, ax = plt.subplots(figsize=(10, 4))
    for year in range(1, df['year'].max() + 1):
        min_idx = df.loc[df['year'] == year].index.min()
        max_idx = df.loc[df['year'] == year].index.max() + 1
        if not year % 2 == 0:
            ax.axvspan(min_idx, max_idx, color='grey', alpha=0.2, lw=0)

        ax.text(0.5 * (max_idx + min_idx) - 3.18, 175, f"Year {year}")

    bar_width = 0.7
    ax.bar(range(len(df)), df['actions'], width=bar_width, color='#0096C7', align='edge', alpha=1,
           edgecolor='k', linewidth=0.5)

    ax_b = ax.twinx()
    ax_b.plot(np.arange(df.index.min() + (bar_width / 2), df.index.max() + (bar_width / 2) + 1, 1), df['cost'],
              c='C1', marker='o', markerfacecolor='none', markersize=5)

    ax.set_xlim(df.index.min() - 2, df.index.max() + 2)
    ax.set_ylabel('Number of investments')
    ax_b.set_ylabel('Cost (€)')
    ax.set_xlabel('Iteration')
    plt.tight_layout()


def tank_volume():
    fig, axes = plt.subplots(nrows=2, ncols=3, sharex=True, sharey=True, figsize=(6, 3))
    axes = axes.ravel()
    for y in range(6):
        wn = all_networks['y' + str(y)]
        sim = wntr.sim.EpanetSimulator(wn)
        results = sim.run_sim()
        tank_vol = results.node['pressure'].loc[:, 'T1_CO']
        axes[y].plot(range(len(tank_vol)), tank_vol)
        axes[y].grid()
        axes[y].set_title(f'Year {int(y)}')

    fig.text(0.5, 0.04, 'Time (hr)', ha='center')
    fig.text(0.04, 0.5, 'Level (m)', va='center', rotation='vertical')
    plt.tight_layout()


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


def export_pipes_first_replacement_year(export_path):
    temp_net = wntr.network.WaterNetworkModel(os.path.join('resources', 'networks', 'BIWS.inp'))  # raw network
    df = pd.read_csv(os.path.join('output', 'pipes_repair_year.csv'), index_col=0)
    all_pipes = temp_net.query_link_attribute(attribute='length', link_type=wntr.network.Pipe).rename('length')
    all_pipes = pd.merge(all_pipes, df, left_index=True, right_on='pipe', how='outer')
    all_pipes = all_pipes.fillna(0)
    all_pipes.set_index('pipe', inplace=True)
    all_pipes = all_pipes.drop('length', axis=1)
    all_pipes.to_csv(export_path)


def supply_vs_demand():
    df = pd.DataFrame()
    for y in range(6):
        evaluator = Evaluator([all_networks['y'+str(y)]])
        evaluator.run_hyd()
        expected_dem = evaluator.expected_demand.sum(axis=1) * 1000
        actual_dem = evaluator.results_demand[evaluator.dem_nodes.index]
        actual_dem = np.where(actual_dem < 0, 0, actual_dem)
        actual_dem = actual_dem.sum(axis=1) * 1000
        df = pd.concat([df, pd.DataFrame({'Expected': expected_dem.sum(), 'Actual': actual_dem.sum()}, index=[y])])

    df.plot(kind='bar', stacked=True, color=COLORS, edgecolor='k', width=0.6, figsize=(8, 4), linewidth=0.6)


def pareto_plot(n):
    def add_pareto_step(ax, data_col, portion):
        mask = data_col >= portion
        min_index = data_col.loc[mask].idxmin()
        min_row = data_col.loc[min_index]
        ax.hlines(y=portion, xmin=min_index, xmax=n, linestyle="--", color='k', linewidth=1)
        ax.vlines(x=min_index, ymin=0, ymax=portion, linestyle="--", color='k', linewidth=1)
        t = ax.text(n-100, portion - 11, f"{min_index} leakages\n{portion}% of total vol")
        t.set_bbox(dict(facecolor='w', alpha=0.5, edgecolor='none'))

        return ax

    df = pd.read_csv("all_leaks_summary.csv", index_col=0)
    df = df.sort_values("total_water_loss", ascending=False)
    df.reset_index(inplace=True)
    df["proportional_loss"] = 100 * df['total_water_loss'] / df['total_water_loss'].sum()
    df["cumm_loss"] = df["proportional_loss"].cumsum()

    fig, ax1 = plt.subplots(figsize=(12, 4))
    ax1.bar(range(len(df)), df['total_water_loss'], width=1, alpha=0.4)

    ax2 = ax1.twinx()
    ax2.plot(range(len(df)), df["cumm_loss"], 'r')

    ax2 = add_pareto_step(ax2, df['cumm_loss'], 30)
    ax2 = add_pareto_step(ax2, df['cumm_loss'], 50)
    ax2 = add_pareto_step(ax2, df['cumm_loss'], 80)

    ax1.set_xlim(-10, n)
    ax1.set_ylim(-1, 1650)
    ax2.set_ylim(0, 102)

    ax1.set_xlabel("Leakage")
    ax1.set_ylabel('Total water loss at year 0 ($m^{3})$')
    ax2.set_ylabel('Cumulative % of total loss')


def percentage_pareto_plot():
    def plot_pareto_step(ax, values, lb, ub):
        ax.hlines(y=min(values[:lb]), xmin=lb, xmax=ub + 1, linestyle="--", color='k', linewidth=1)
        ax.vlines(x=ub + 1, ymin=pareto.loc[ub, 'percentage_loss'], ymax=min(values[:lb]), linestyle="--", color='k',
                  linewidth=1)

        leakges_num = pareto.loc[ub + 1, 'bounds'] - pareto.loc[lb, 'bounds']
        try:
            volume = pareto.loc[ub, 'cum_percentage'] - pareto.loc[lb - 1, 'cum_percentage']
        except KeyError:
            volume = pareto.loc[ub, 'cum_percentage']

        t = ax.text(lb + 1, min(values[:lb]) + 200, f"{leakges_num} leakages,\n{volume:.1%} volume")
        return ax

    df = pd.read_csv("all_leaks_summary.csv", index_col=0)
    df = df.sort_values('total_water_loss', ascending=False)
    df.reset_index(inplace=True)
    df['cum_loss'] = df['total_water_loss'].cumsum()
    total_loss = df['total_water_loss'].sum()

    pareto = pd.DataFrame(index=range(1, 101))
    pareto['bounds'] = [int(np.ceil(0.01 * i * len(df))) for i in range(1, 101)]
    pareto = pd.merge(pareto, df['cum_loss'], left_on='bounds', right_index=True, how='left')
    pareto['cum_percentage'] = pareto['cum_loss'] / total_loss
    pareto['percentage_loss'] = pareto['cum_loss'] - pareto['cum_loss'].shift()
    pareto.loc[1, 'percentage_loss'] = pareto['cum_loss'].iloc[0]

    fig, ax1 = plt.subplots(figsize=(9, 4))
    ax1.bar(range(1, 101), pareto['percentage_loss'], alpha=0.5, align='edge', edgecolor='k')
    ax1.set_ylabel('Cumulative Loss ($m^{3})$')

    ax1.set_xlabel('Portion from all leakages (%)')
    ax1 = plot_pareto_step(ax1, pareto['percentage_loss'], 1, 5)
    ax1 = plot_pareto_step(ax1, pareto['percentage_loss'], 6, 20)

    ax2 = ax1.twinx()
    ax2.plot(range(1, 101), pareto['cum_percentage'], 'r')

    ax2.set_ylabel('Cumulative normalized Loss (%)', color='r')
    for tl in ax2.get_yticklabels():
        tl.set_color('r')

    ax1.set_ylim(0, max(pareto['percentage_loss']) * 1.15)
    ax1.set_xlim(-1, 101)


def compare_two_solutions(sol1, sol2):
    df1 = pd.read_csv(sol1, index_col=0)
    df2 = pd.read_csv(sol2, index_col=0)

    mat = (df2 - df1) / df2
    mat.iloc[:, [2, 6, 7]] *= -1
    vmin, vmax = -1, 1

    fig, ax = plt.subplots()
    cax = ax.matshow(mat, cmap='bwr', vmin=vmin, vmax=vmax)

    divider = make_axes_locatable(ax)
    cax1 = divider.append_axes("right", size="5%", pad=0.1)

    cbar = plt.colorbar(cax, cax=cax1, orientation='vertical')
    cbar.set_ticks([vmin, vmax])
    cbar.set_ticklabels(['Perelman et al.', 'Marsili et al.'], rotation=-45, va='center')

    ax.tick_params(labelbottom=True, labeltop=False, top=False)
    ax.set_xticks(np.arange(mat.shape[1]))
    ax.set_yticks(np.arange(mat.shape[0]))
    ax.set_xticklabels(['1', '2', '3', '4', '5', '6', '7', '8', '9'])
    ax.set_yticklabels(['0', '1', '2', '3', '4', '5', 'Total'])

    ax.set_xticks(np.arange(-.5, mat.shape[1] - 0.5, 1), minor=True)
    ax.set_yticks(np.arange(-.5, mat.shape[0] - 0.5, 1), minor=True)
    ax.grid(which='minor', color='k', linestyle='-')

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            text = ax.text(j, i, f"{mat.iloc[i, j]:.1%}", ha="center", va="center", color="k", fontsize=9)

    ax.set_xlabel('Indicator')
    ax.set_ylabel('Year')


if __name__ == "__main__":
    y1_greedy = os.path.join('output', '1_greedy_output', '20230606080239_y1')
    y2_greedy = os.path.join('output', '1_greedy_output', '20230608092158_y2')
    y3_greedy = os.path.join('output', '1_greedy_output', '20230610160100_y3')
    y4_greedy = os.path.join('output', '1_greedy_output', '20230611104545_y4')
    y5_greedy = os.path.join('output', '1_greedy_output', '20230613112601_y5')
    all_greedy = [y1_greedy, y2_greedy, y3_greedy, y4_greedy, y5_greedy]

    SOLUTION_PATH = os.path.join('output/8_final_networks_adjusted_new_controls')
    all_networks = load_networks()
    all_pipes = pd.read_csv('resources/pipes_data.csv')

    metrics_by_year('output/8_final_networks_adjusted_new_controls/score.csv')  # Figure 5

    compare_two_solutions('output/8_final_networks_adjusted_new_controls/score.csv',
                          'output/marsili_et_al_score.csv')  # Figure 6

    # plot_leaks(all_networks, leaks_summary_path='resources/all_leaks_summary.csv')  # Figure 7

    # Figure 8 - done with GIS
    # export_pipes_first_replacement_year(os.path.join('output'', 'pipes_repair_year.csv'))

    plot_evaluations_per_pipe()  # Figure 9
    plot_iterations('output/all_iter.csv')  # Figure 10

    tank_volume()  # Figure S2
    plt.show()
