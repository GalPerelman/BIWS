import os
import glob
import numpy as np
import pandas as pd
import wntr

import utils

pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.set_option("display.precision", 3)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESOURCES_DIR = os.path.join(BASE_DIR, 'resources')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')


class BuildSolution:
    def __init__(self, base_networks_path: str, solution_file: str, output_dir, cumulative_budget: dict):
        """

        base_networks_path:     path to directory with networks after setting emitters coefficients by year
                                and before any investment was made
        solution_file:          path to a scv file with all the investments
        output_dir:             directory to export results - networks and new csv
        cumulative_budget:      a dictionary with the available budget per year {0: 0, 1: 470000, 2: 1120, 3: 1770 ...}
        """
        self.base_networks_path = base_networks_path
        self.solution_path = solution_file
        self.output_dir = utils.validate_dir_path(output_dir)
        self.cumulative_budget = cumulative_budget

        self.networks_by_year = self.build_base_networks()
        self.solution = pd.read_csv(self.solution_path)
        self.all_pipes = self.get_pipes_data()
        self.all_leaks = pd.read_csv(os.path.join(RESOURCES_DIR, "preprocess", "leaks_preprocess.csv"))

    def build_base_networks(self):
        networks = {}
        for i, file_path in enumerate(glob.glob(os.path.join(self.base_networks_path, '*.inp'))):
            file_name = os.path.basename(file_path)
            if file_name[1].isdigit():
                year = int(file_name[1])
                networks[year] = wntr.network.WaterNetworkModel(os.path.join(self.base_networks_path, file_name))
            else:
                raise Exception("base networks files names not in format")

        return networks

    def get_pipes_data(self):
        all_pipes = pd.DataFrame(self.networks_by_year[0].query_link_attribute('diameter'), columns=['diameter'])
        all_pipes['original_id'] = all_pipes.index.to_series().str.split('_').str[0]
        return all_pipes

    def replace_single_pipe(self, net, pipe_id, diameter_m):
        pipe_segments = self.all_pipes.loc[self.all_pipes['original_id'] == pipe_id]
        pipe_leaks = self.all_leaks.reset_index().merge(pipe_segments, left_on='Link', right_index=True)
        pipe_leaks = pipe_leaks.drop_duplicates().set_index('Leak_id')

        cost = 0
        if pipe_segments.empty:
            pipe = net.get_link(pipe_id)
            pipe.diameter = diameter_m
            pipe.roughness = 120
            cost += utils.replace_pipe_cost(diameter_m, pipe.length)
        else:
            for i, row in pipe_leaks.iterrows():
                net, leak_cost = self.repair_single_leak(net, row.name, row['diameter'] * 1000)
                #  ignoring the cost of fixing leaks when replacing the pipe
            for i, row in pipe_segments.iterrows():
                pipe = net.get_link(row.name)
                pipe.diameter = diameter_m
                pipe.roughness = 120
                cost += utils.replace_pipe_cost(diameter_m, pipe.length)

        return net, cost

    def repair_single_leak(self, net, leak_id, pipe_diameter_mm):
        try:
            leak = net.get_node(leak_id)
            coef = leak.emitter_coefficient * 1000  # wntr uses only SI units, converting to (l/s)/m
            cost = utils.fix_leak_cost(coef, pipe_diameter_mm)

            net.remove_link('LeakPipe_' + leak_id.split('_')[1])
            net.remove_node(leak_id)
        except KeyError as e:
            cost = 0
        return net, cost

    def get_pipe_cost(self, net, pipe_id, diameter_m):
        pipe_segments = self.all_pipes.loc[self.all_pipes['original_id'] == pipe_id]
        cost = 0
        if pipe_segments.empty:
            pipe = net.get_link(pipe_id)
            cost += utils.replace_pipe_cost(diameter_m, pipe.length)
        else:
            for i, row in pipe_segments.iterrows():
                pipe = net.get_link(row.name)
                cost += utils.replace_pipe_cost(diameter_m, pipe.length)
        return cost

    def build(self):
        years_budget = self.cumulative_budget
        years_used_budget = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
        all_actions = pd.DataFrame()

        year = 0
        net = self.networks_by_year[year]

        for i, row in self.solution.iterrows():
            if row['action'] == 'pipe':
                cost = self.get_pipe_cost(net, row['element'], row['new_diameter_m'])
                if years_used_budget[year] + row['d_cost'] < years_budget[year]:
                    net = self.networks_by_year[year]

                else:
                    if year == 5:
                        break
                    else:
                        year += 1
                        net = self.networks_by_year[year]

                for y in range(year, 6):
                    y_net = self.networks_by_year[y]
                    y_net, cost = self.replace_single_pipe(y_net, row['element'], row['new_diameter_m'])
                years_used_budget[year] = years_used_budget[year] + row['d_cost']
                row['new_cost'] = row['d_cost']
                row['new_year'] = year
                all_actions = pd.concat([all_actions, pd.DataFrame(row).T])

            elif row['action'] == 'leak':
                try:
                    # check if leak is already repaired by replacing its pipe
                    leak_coef = net.get_node(row['element']).emitter_coefficient * 1000
                except KeyError:
                    continue

                cost = utils.fix_leak_cost(leak_coef, row['leak_pipe_diameter_mm'])
                if years_used_budget[year] + cost < years_budget[year]:
                    net = self.networks_by_year[year]

                else:
                    if year == 5:
                        break
                    else:
                        year += 1
                        net = self.networks_by_year[year]

                for y in range(year, 6):
                    y_net = self.networks_by_year[y]
                    y_net, cost = self.repair_single_leak(y_net, row['element'], row['leak_pipe_diameter_mm'])

                years_used_budget[year] = years_used_budget[year] + cost
                row['new_cost'] = cost
                row['new_year'] = year
                all_actions = pd.concat([all_actions, pd.DataFrame(row).T])

        all_actions.to_csv(os.path.join(self.output_dir, "all_actions.csv"))
        for year, net in self.networks_by_year.items():
            wntr.network.io.write_inpfile(net, os.path.join(self.output_dir, "y" + str(year) + ".inp"))


def get_solution_dir_names(greedy_output):
    names = {}
    for file in os.listdir(greedy_output):
        d = os.path.join(greedy_output, file)
        if os.path.isdir(d):
            names[os.path.basename(d).split('_')[1]] = d

    return names


def clean_results(year: str):
    """
    Input is 'y1', 'y2'...
    Clean duplicated actions from the list of investments
    Two types of 'duplications':
    1) Pipes that are enlarged multiple times in the same year (in different iterations of the greedy)
    2) Leaks that are repaired and in later iteration (of the same year) the pipe of the leak is replaced

    These two cases can generate also duplications between different years.
    For example a leak is repaired in y1 and then the leak pipe is replaced in y2
    During the battle these cases were analyzed and a conclusion that removing duplications across different years
    is not necessarily improve the results.
    """
    path = solution_dir_names[year]
    df = pd.read_csv(os.path.join(path, year + '_actions-finalized.csv'))
    pipes = df.loc[df['action'] == 'pipe']
    leaks = df.loc[df['action'] == 'leak']

    # step 1 - check for duplicate pipes
    pipes = pipes.sort_values(by='new_diameter_m')
    pipes = pipes.drop_duplicates(subset='element', keep="last")

    # add pipes length
    pipes_data = pd.read_csv(os.path.join(RESOURCES_DIR, 'pipes_data.csv'))
    pipes = pd.merge(pipes, pipes_data[['ID', 'Length']], left_on='element', right_on='ID')

    # step 2 - check for leaks repairs in replaced pipes
    leaks_data = pd.read_csv(os.path.join(RESOURCES_DIR, 'preprocess', 'leaks_preprocess.csv'))
    leaks = pd.merge(leaks, leaks_data[['Leak_id', 'Link']], left_on='element', right_on='Leak_id')
    repaired_leaks = leaks.loc[~leaks['Link'].isin(pipes['element'])]
    replaced_pipes_leaks = leaks.loc[leaks['Link'].isin(pipes['element'])]

    actions = pd.concat([pipes, leaks], axis=0)
    all_repaired_leaks = pd.concat([repaired_leaks, replaced_pipes_leaks], axis=0)
    return actions, pipes, all_repaired_leaks, repaired_leaks, replaced_pipes_leaks


if __name__ == "__main__":
    final_netwotks = os.path.join(OUTPUT_DIR, '2_fcv', '2_final_networks')
    greedy_output = os.path.join(OUTPUT_DIR, '2_fcv', '1_greedy_output')
    solution_dir_names = get_solution_dir_names(greedy_output)
    # get_all_actions()
    # for y in ['y1', 'y2', 'y3', 'y4', 'y5']:
    #     actions, pipes, all_repaired_leaks, repaired_leaks, replaced_pipes_leaks = clean_results(y)
    #     print(y, len(repaired_leaks), len(replaced_pipes_leaks))

    sol_builder = BuildSolution(base_networks_path=os.path.join(RESOURCES_DIR, 'networks', 'Base-Pumps'),
                                solution_file=os.path.join('output', 'fcv', 'all_actions_and_optional_completions.csv'),
                                output_dir=os.path.join('output', 'fcv', '5_final_networks_adjusted'),
                                cumulative_budget={0: 0, 1: 472917, 2: 650000, 3: 650000, 4: 650000, 5: 650000}
                                )

    sol_builder.build()
