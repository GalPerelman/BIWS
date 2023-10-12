import os
import glob
import json
import warnings
import pandas as pd
import itertools
from itertools import combinations
from itertools import product
from copy import deepcopy
from tqdm import tqdm
import wntr
import wntr.network.controls as controls

import utils
from metrics import Evaluator

warnings.filterwarnings(action='ignore', module='wntr')
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESOURCES_DIR = os.path.join(BASE_DIR, 'resources')


class ExhaustiveSearch:
    def __init__(self, inp_path, zone_name, valves_names, output_dir):
        self.inp_path = inp_path
        self.zone_name = zone_name
        self.valves_names = valves_names
        self.output_dir = output_dir

        self.inp_name = utils.get_file_name_from_path(self.inp_path)[0]
        self.output_dir = utils.validate_dir_path(os.path.join(self.output_dir))
        self.wn = wntr.network.WaterNetworkModel(self.inp_path)
        self.wn = set_all_valves_active(self.wn)
        self.wn = clear_group_controls(self.wn)
        self.build_search_space()
        self.results = {}

    def build_search_space(self):
        all_configs = list(product([0, 1], repeat=len(self.valves_names)))
        all_combinations = list(product(all_configs, repeat=len(['day', 'night'])))
        df = pd.DataFrame(columns=[(x, y) for x in self.valves_names for y in ['day', 'night']],
                          data=[list(_[0]) + list(_[1]) for _ in all_combinations])
        return df

    def search(self):
        controls_clock = {'day': 7, 'night': 0}
        controls_status = {0: 'close', 1: 'open'}

        df = self.build_search_space()
        results = pd.DataFrame()
        for i, row in tqdm(df.iterrows(), total=len(df)):
            temp_net = deepcopy(self.wn)
            for (v_name, period) in df.columns:
                valve = temp_net.get_link(v_name)
                status = row[(v_name, period)]
                start_time = controls_clock[period]  # time when control start
                condition = controls.TimeOfDayCondition(temp_net, relation='=', threshold=start_time * 3600)
                control = controls.Control(condition, controls.ControlAction(valve, 'status', status))
                temp_net.add_control(f'str{v_name}-{period}-{controls_status[status]}', control)

            evaluator = Evaluator([temp_net])
            metrics = utils.round_dict(evaluator.evaluate_scenario(), 4)
            results = pd.concat([results, pd.DataFrame(metrics, index=[i])])

        df = pd.merge(df, results, left_index=True, right_index=True)
        df.to_csv(os.path.join(self.output_dir, f"{self.inp_name}-{self.zone_name}-final.csv"))


def set_all_valves_active(wn: wntr.network.WaterNetworkModel):
    for v_name, valve in wn.valves():
        valve._status = "Active"
    return wn


def clear_group_controls(wn: wntr.network.WaterNetworkModel):
    """
    This function clear the existing controls for ALL valves
    """
    controls_to_remove = []
    for cont_name, cont in wn.controls():
        controlled_element = list(cont.requires())[-1]
        if controlled_element.name in wn.valve_name_list:
            controls_to_remove.append(cont_name)

    for cont_name in controls_to_remove:
        wn.remove_control(cont_name)

    return wn


def get_best_config(file_path):
    df = pd.read_csv(file_path, index_col=0)
    df = df.rename(columns={col: int(col) for col in ['1', '2', '3', '4', '5', '6', '7', '8', '9']})

    best = {1: df[1].max(), 2: df[2].max(), 3: df[3].min(), 4: df[4].max(), 5: df[5].max(), 6: df[6].max(),
            7: df[7].min(), 8: df[8].min(), 9: df[9].max()}

    worst = {1: df[1].min(), 2: df[2].min(), 3: df[3].max(), 4: df[4].min(), 5: df[5].min(), 6: df[6].min(),
             7: df[7].max(), 8: df[8].max(), 9: df[9].min()}

    df['score'] = sum([(df[_] - worst[_]) / (best[_] - worst[_]) for _ in range(1, 10)])
    df = df.sort_values('score', ascending=False)

    best_config = df.nlargest(1, columns='score')
    best_config = best_config.drop(['score'] + [_ for _ in range(1, 10)], axis=1)
    return best_config


def set_config_to_net(wn, config):
    controls_clock = {'day': 7, 'night': 0}
    controls_status = {0: 'close', 1: 'open'}

    for col in config.columns:
        values = col[1:-1].split(', ')
        values = [x[1:-1] if x.startswith("'") and x.endswith("'") else x for x in values]
        actual_tuple = tuple(values)
        valve_name, period = actual_tuple

        valve = wn.get_link(valve_name)
        status = config[col].values[0]

        start_time = controls_clock[period]  # time when control start
        condition = controls.TimeOfDayCondition(wn, relation='=', threshold=start_time * 3600)
        control = controls.Control(condition, controls.ControlAction(valve, 'status', status))
        try:
            # if a control for the same valve and period is already set (from a different cluster)
            # print the controls so user can check there are no conflicts
            wn.add_control(f'{valve_name}-{period}-{controls_status[status]}', control)
        except ValueError:
            print(f'{valve_name}-{period}-{controls_status[status]}')

    return wn


def write_all_best_cfg(controls_output_dir: str, base_networks_dir: str, output_dir: str):
    for y in range(6):
        inp_path = os.path.join(base_networks_dir, f'y{y}.inp')
        wn = wntr.network.WaterNetworkModel(inp_path)
        year_controls_dir = os.path.join(controls_output_dir, f'y{y}') + "/*.csv"
        for file in glob.glob(year_controls_dir):
            cfg = get_best_config(file)
            wn = set_config_to_net(wn, cfg)

        output_net_path = os.path.join(output_dir, f'y{y}.inp')
        wntr.network.io.write_inpfile(wn, output_net_path)


if __name__ == "__main__":
    for y in range(6):
        inp_path = os.path.join('output', '5_final_networks_post', f'y{y}.inp')
        valves_file_path = os.path.join(RESOURCES_DIR, 'valves.json')
        output_path = os.path.join('output', '3_controls_search', f'y{y}')
        with open(valves_file_path) as f:
            valves = json.load(f)

            for cluster, valves_names in valves.items():
                print("==========", 'y', y, '-', cluster, "==========")
                es = ExhaustiveSearch(inp_path, cluster, valves_names, output_path)
                es.search()

    write_all_best_cfg(os.path.join('output', 'fcv', '3_controls_search'),
                       os.path.join('output', 'fcv', '5_final_networks_post'),
                       os.path.join('output', 'fcv', '7_final_networks_post_controls'))
