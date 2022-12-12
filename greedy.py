import os
import pandas as pd
import numpy as np
import wntr
import time
from tqdm import tqdm
from copy import deepcopy

from metrics import Evaluator
import utils

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESOURCES_DIR = os.path.join(BASE_DIR, 'resources')
OUTPUT_DIR = os.path.join(BASE_DIR, "output")


class Greedy:
    def __init__(self, inp_path, output_dir, budget, actions_ratio, hgl_threshold,
                 n_leaks, reevaluate_ratio, total_run_time, load_init_path=False, hours_duration=False):

        self.inp_path = inp_path
        self.output_dir = utils.validate_dir_path(output_dir)
        self.budget = budget
        self.actions_ratio = actions_ratio
        self.hgl_threshold = hgl_threshold          # Threshold for pipes candidates
        self.n_leaks = n_leaks                      # Number of candidates leaks
        self.reevaluate_ratio = reevaluate_ratio
        self.total_run_time = total_run_time        # Greedy run time
        self.load_init_path = load_init_path        # For using last iteration of previous year as input
        self.hours_duration = hours_duration        # Hydraulic simulation duration (if different from inp)

        self.net_name = utils.get_file_name_from_path(self.inp_path)[0]
        self.net = wntr.network.WaterNetworkModel(self.inp_path)

        if self.hours_duration:
            self.net.options.time.duration = 3600 * self.hours_duration

        self.all_pipes = pd.DataFrame(self.net.query_link_attribute('diameter'), columns=['diameter'])
        self.all_pipes['original_id'] = self.all_pipes.index.to_series().str.split('_').str[0]
        self.pipes, self.leaks, self.all_leaks = self.get_candidates()
        self.iter_evaluations, self.iter_budget = self.get_iter_evaluations_and_budget()
        self.detection_record = pd.DataFrame()

    def get_iter_evaluations_and_budget(self):
        n_candidates = len(self.pipes) + len(self.leaks)
        iter_evaluations = int(self.reevaluate_ratio * n_candidates)
        num_iter = self.total_run_time * 3600 / (iter_evaluations * 60)  # 60 sec - estimated single evaluation
        iter_budget = self.budget / num_iter
        return iter_evaluations, iter_budget

    def get_candidates(self):
        evaluator = Evaluator([self.net])
        evaluator.run_hyd()

        pipes = evaluator.pipes_headloss_summary(0)[['ID', 'Length', 'Diameter', 'mean_head_loss']]
        pipes['hgl'] = pipes['mean_head_loss'].abs() / pipes['Length']
        pipes = pipes.sort_values('ID')
        pipes = pipes.rename(columns={'ID': 'link_id'})
        pipes.loc[:, 'current_cost'] = 0
        pipes.loc[:, 'evaluate_flag'] = 1
        pipes = pipes.loc[pipes['hgl'] >= self.hgl_threshold]  # Select candidates
        pipes.sort_values('hgl', inplace=True, ascending=False)
        pipes.set_index('link_id', inplace=True)

        leaks = evaluator.leaks_summary()
        leaks.loc[:, 'evaluate_flag'] = 1
        leaks.set_index('Leak_id', inplace=True)
        all_leaks = deepcopy(leaks)
        leaks['rank'] = leaks['total_water_loss'] / leaks['total_cost']
        leaks.sort_values('rank', inplace=True, ascending=False)
        leaks = leaks.nlargest(self.n_leaks, 'rank')
        return pipes, leaks, all_leaks

    def replace_single_pipe(self, net, pipe_id, diameter_m):
        pipe_segments = self.all_pipes.loc[self.all_pipes['original_id'] == pipe_id]
        pipe_leaks = self.all_leaks.reset_index().merge(pipe_segments, left_on='Link', right_on='original_id') \
            .drop_duplicates().set_index('Leak_id')
        cost = 0
        if pipe_segments.empty:
            pipe = net.get_link(pipe_id)
            pipe.diameter = diameter_m
            pipe.roughness = 120
            cost += utils.replace_pipe_cost(diameter_m, pipe.length)
        else:
            for i, row in pipe_leaks.iterrows():
                net, leak_cost = self.repair_leak(net, row.name, row['diameter'] * 1000)
                #  ignoring the cost of fixing leaks when replacing the pipe

            for i, row in pipe_segments.iterrows():
                pipe = net.get_link(row.name)
                pipe.diameter = diameter_m
                pipe.roughness = 120
                cost += utils.replace_pipe_cost(diameter_m, pipe.length)

        return net, cost

    def repair_leak(self, func_net, leak_id, pipe_diameter_mm):
        try:
            leak = func_net.get_node(leak_id)
            coef = leak.emitter_coefficient * 1000  # wntr uses only SI units, converting to (l/s)/m
            cost = utils.fix_leak_cost(coef, pipe_diameter_mm)
            # leak.emitter_coefficient = 0
            func_net.remove_link('LeakPipe_' + leak_id.split('_')[1])
            func_net.remove_node(leak_id)
        except KeyError:
            cost = 0
        return func_net, cost

    def get_pipes_marginal_utility(self, net, worse_benchmark):
        df = pd.DataFrame(columns=['action', 'element', 'new_diameter_m', 'd_cost', 'so'])
        pipes_to_evaluate = self.pipes.loc[self.pipes['evaluate_flag'] == 1]
        for i, row in tqdm(pipes_to_evaluate.iterrows(), total=len(pipes_to_evaluate), desc='pipes'):
            temp_net = deepcopy(net)
            current_diameter_mm = temp_net.get_link(row.name).diameter * 1000
            if current_diameter_mm == 800:
                scenario = pd.DataFrame({'action': 'pipe', 'element': row.name, 'new_diameter_m': 10 ** 5,
                                         'd_cost': 10 ** 5, 'so': 9}, index=[i])
            else:
                next_diam_m = utils.get_next_diameter(current_diameter_mm) / 1000

                temp_net, cost = self.replace_single_pipe(temp_net, row.name, next_diam_m)
                delta_cost = cost - self.pipes.loc[self.pipes.index == row.name, 'current_cost'].values[0]
                obj, flow, pressure = self.get_objectives_and_detect_changes(temp_net)
                so = sum(utils.normalize_obj(worse_benchmark, obj).values())  # Normalized single objective
                scenario = pd.DataFrame({'action': 'pipe', 'element': row.name, 'new_diameter_m': next_diam_m,
                                         'd_cost': delta_cost, 'so': so}, index=[i])
                scenario = pd.concat([scenario, pd.DataFrame(obj, index=[i])], axis=1)
            df = pd.concat([df, scenario], ignore_index=True)
        return df

    def get_leaks_marginal_utility(self, net, worse_benchmark):
        df = pd.DataFrame(columns=['action', 'element', 'leak_pipe_diameter_mm', 'd_cost', 'so'])
        leaks_to_evaluate = self.leaks.loc[self.leaks['evaluate_flag'] == 1]
        for i, row in tqdm(leaks_to_evaluate.iterrows(), total=len(leaks_to_evaluate), desc="leaks"):
            temp_net = deepcopy(net)
            diameter_mm = temp_net.get_link(row['Link_id']).diameter * 1000
            temp_net, cost = self.repair_leak(temp_net, row.name, diameter_mm)
            obj, flow, pressure = self.get_objectives_and_detect_changes(temp_net)
            so = sum(utils.normalize_obj(worse_benchmark, obj).values())  # Normalized single objective
            scenario = pd.DataFrame({'action': 'leak', 'element': row.name, 'leak_pipe_diameter_mm': diameter_mm,
                                     'd_cost': cost, 'so': so}, index=[i])
            scenario = pd.concat([scenario, pd.DataFrame(obj, index=[i])], axis=1)
            df = pd.concat([df, scenario], ignore_index=True)
        return df

    def estimate_marginal_benefit(self, net, current_state):
        pipes = self.get_pipes_marginal_utility(net, current_state)
        leaks = self.get_leaks_marginal_utility(net, current_state)

        df = pd.concat([pipes, leaks], axis=0)
        df['d_benefit'] = 9 - df['so']
        if df['d_cost'].eq(0).any().any():
            df['d_cost'] = np.where(df['d_cost'] == 0, 10 ** -5, df['d_cost'])
        df['mb'] = df['d_benefit'] / df['d_cost']
        return df.sort_values('mb', ascending=False)

    def get_evaluation_flag(self, improved_net, benchmark_flows):
        obj, flows, pressures = self.get_objectives_and_detect_changes(improved_net)
        df = pd.merge(benchmark_flows.rename('benchmark'), flows, left_index=True, right_index=True)
        df = df.loc[df.index.isin(self.pipes.index.tolist() + self.leaks['Link_id'].tolist())]
        df['delta'] = np.abs(df['benchmark'] - df['flow'])
        self.detection_record = pd.concat([self.detection_record, df['delta']], axis=1)
        self.detection_record.to_csv(os.path.join(self.output_dir, 'detection_monitoring.csv'))

        df['evaluate_flag'] = np.where(df.index.isin(df.nlargest(self.iter_evaluations, 'delta').index), 1, 0)
        self.pipes = pd.merge(self.pipes.loc[:, self.pipes.columns != 'evaluate_flag'], df['evaluate_flag'],
                              left_index=True, right_index=True)
        self.leaks = pd.merge(self.leaks.loc[:, self.leaks.columns != 'evaluate_flag'], df['evaluate_flag'],
                              left_on='Link_id', right_index=True)

        return obj

    def improve_net(self, net, actions: pd.DataFrame):
        improve_cost = 0
        for i, row in actions.iterrows():
            if row['action'] == 'pipe':
                net, cost = self.replace_single_pipe(net, row.name, row['new_diameter_m'])
                delta_cost = cost - self.pipes.loc[self.pipes.index == row.name, 'current_cost'].values[0]
                self.pipes.loc[self.pipes.index == row.name, 'current_cost'] = cost
                leaks_for_pipe = self.leaks.loc[self.leaks['Link'] == row.name]
                self.leaks = self.leaks.drop(self.leaks[self.leaks.index.isin(leaks_for_pipe.index)].index)
                improve_cost += delta_cost
            elif row['action'] == 'leak':
                net, cost = self.repair_leak(net, row.name, row['leak_pipe_diameter_mm'])
                self.leaks = self.leaks.drop(self.leaks[self.leaks.index == row.name].index, axis=0)
                self.all_leaks = self.all_leaks.drop(self.all_leaks[self.all_leaks.index == row.name].index, axis=0)
                improve_cost += cost

        return net, improve_cost

    def get_objectives_and_detect_changes(self, net):
        try:
            evaluator = Evaluator([net])
            benchmark = evaluator.evaluate_scenario()
            flows = np.abs(evaluator.results_flow).mean().rename('flow')
            pressures = evaluator.results_pressure.mean().rename('pressure')
        except Exception as e:
            wntr.network.io.write_inpfile(net, os.path.join(self.output_dir, 'debuging.inp'))
            print(e)
        return benchmark, flows, pressures

    def start(self):
        used_budget = 0
        n_iter = 1
        x_net = deepcopy(self.net)
        results = pd.DataFrame()
        iterations = pd.DataFrame()
        start_time = time.time()
        break_flag = False

        while used_budget < self.budget:
            iter_start_time = time.time()
            benchmark, flows, pressures = self.get_objectives_and_detect_changes(x_net)

            if n_iter == 1 and self.load_init_path:
                self.pipes = pd.read_csv(os.path.join(self.load_init_path, "pipes_flags_1.csv"), index_col=0)
                self.leaks = pd.read_csv(os.path.join(self.load_init_path, "leaks_flags_1.csv"), index_col='Leak_id')
                new_evaluations = pd.read_csv(os.path.join(self.load_init_path, "input_1.csv"), index_col='element')
            else:
                new_evaluations = self.estimate_marginal_benefit(x_net, benchmark)
                new_evaluations.set_index('element', inplace=True)
                new_evaluations.columns = [str(col) for col in new_evaluations.columns]

            if n_iter == 1:
                evaluations = deepcopy(new_evaluations)

            num_eval = self.pipes['evaluate_flag'].sum() + self.leaks['evaluate_flag'].sum()
            evaluations = new_evaluations.combine_first(evaluations).sort_values('mb', ascending=False)

            actions = evaluations.loc[evaluations['mb'] >= evaluations['mb'].max() * (1 - self.actions_ratio)]

            iter_cost = actions['d_cost'].sum()
            if iter_cost <= self.iter_budget:
                actions = evaluations[evaluations['d_cost'].cumsum() <= self.iter_budget]
            if used_budget + actions['d_cost'].sum() > self.budget:
                actions = actions[actions['d_cost'].cumsum() < self.budget - used_budget]
                break_flag = True

            num_actions = len(actions)
            actions['iter'] = str(n_iter)
            results = pd.concat([results, actions], axis=0)
            x_net, cost = self.improve_net(x_net, actions)
            used_budget += cost

            self.pipes.to_csv(os.path.join(self.output_dir, 'pipes_flags_' + str(n_iter) + '.csv'))
            self.leaks.to_csv(os.path.join(self.output_dir, 'leaks_flags_' + str(n_iter) + '.csv'))
            updated_obj = self.get_evaluation_flag(x_net, flows)

            evaluations.to_csv(os.path.join(self.output_dir, self.net_name + '_' + str(n_iter) + '.csv'))
            x_net.write_inpfile(os.path.join(self.output_dir, self.net_name + '_' + str(n_iter) + '.inp'))

            iter = pd.concat([pd.DataFrame({'time': time.time() - iter_start_time,
                                            'evaluations': num_eval,
                                            'actions': num_actions,
                                            'cost': cost,
                                            'used budget': used_budget}, index=[0]),
                              pd.DataFrame(round_obj(updated_obj, 4), index=[0])], axis=1)

            iterations = pd.concat([iterations, iter])
            print(time.time() - iter_start_time, num_eval, num_actions, cost, used_budget, round_obj(updated_obj, 4))
            evaluations = evaluations[~evaluations.index.isin(actions.index)]

            iterations.to_csv(os.path.join(self.output_dir, self.net_name + '_iterations.csv'))
            results.to_csv(os.path.join(self.output_dir, self.net_name + '_actions.csv'))
            with open(os.path.join(self.output_dir, 'run_details.txt'), 'w') as file:
                file.write('Run time: %0.2f \n' % (time.time() - start_time))
                for k in ['hours_duration', 'inp_path', 'budget', 'actions_ratio', 'hgl_threshold', 'n_leaks',
                          'reevaluate_ratio', 'total_run_time', 'load_init_path']:
                    file.write("{}: {}\n".format(k, self.__dict__[k]))

            n_iter += 1
            if break_flag:
                break

        utils.remove_files('temp.bin', 'temp.inp', 'temp.rpt')


def round_obj(obj, digits):
    return {k: round(v, digits) for k, v in obj.items()}


if __name__ == "__main__":
    output_path = os.path.join(OUTPUT_DIR, time.strftime("%Y%m%d%H%M%S") + '_test')
    # first_iter_results_path = os.path.join(RESOURCES_DIR, "year1_init")
    inp = os.path.join(RESOURCES_DIR, "networks", "BIWS_y0.inp")
    greedy = Greedy(inp,
                    output_dir=output_path,
                    budget=50000,
                    actions_ratio=0.3,
                    hgl_threshold=0.01,
                    n_leaks=10,
                    reevaluate_ratio=0.01,
                    total_run_time=1,
                    hours_duration=6)

    greedy.start()