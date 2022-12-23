import os
import glob
import json
import warnings
import pandas as pd
import itertools
from itertools import combinations
from collections import Counter
from copy import deepcopy
from tqdm import tqdm
import wntr
import wntr.network.controls as controls

import utils
from metrics import Evaluator

warnings.filterwarnings(action='ignore', module='wntr')
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESOURCES_DIR = os.path.join(BASE_DIR, 'resources')


class ControlChecker:
    """
    inp must be cleaned of previous controls

    """
    def __init__(self, inp_path, zone, valves, output_dir):
        self.inp_path = inp_path
        self.zone = zone
        self.valves = valves
        self.output_dir = output_dir

        self.inp_name = utils.get_file_name_from_path(self.inp_path)[0]
        self.output_dir = utils.validate_dir_path(os.path.join(self.output_dir))
        self.wn = wntr.network.WaterNetworkModel(self.inp_path)
        self.wn.options.time.duration = 48*3600

        self.benchmark = Evaluator([self.wn]).evaluate_scenario()
        self.results = {}

    def set_all_elements(self, net, status: str):
        for v_name in self.valves:
            net.get_link(v_name).initial_status = status
        return net

    def record_results(self, desc, metrics, normalized, total_demand, total_supply):
        self.results[desc] = metrics
        self.results[desc]['normalized'] = normalized
        self.results[desc]['total_demand'] = total_demand
        self.results[desc]['total_supply'] = total_supply

    def evaluate_controls(self):
        # all closed
        net = deepcopy(self.wn)
        self.set_all_elements(net, 'Closed')
        metrics, total_demand, total_supply = evaluate(self.wn)
        norm = sum(utils.normalize_obj(self.benchmark, metrics).values())
        self.record_results('all_closed', metrics, norm, total_demand, total_supply)

        # iterate all combinations
        valves_two_settings = [(valve, time) for valve in self.valves for time in ['night', 'morning']]

        for n in range(1, len(valves_two_settings)):
            print(f'analyzing {n} pipe open\n')

            for cvs in tqdm(list(combinations(valves_two_settings, n))):
                net = deepcopy(self.wn)

                # check if both night and morning are open, if yes, no need to add rule
                cv_list = [cv for (cv, _) in cvs]
                cv_count = Counter(cv_list)
                all_open_cv = [cv for cv, count in cv_count.items() if count == 2]

                for cv in all_open_cv:
                    net.get_link(cv).initial_status = 'Open'
                for (cv, time) in cvs:
                    if cv not in all_open_cv:
                        net = self.add_control(net, net.get_link(cv), time)
                metrics, total_demand, total_supply = evaluate(net)
                norm = sum(utils.normalize_obj(self.benchmark, metrics).values())
                self.record_results(cvs, metrics, norm, total_demand, total_supply)

        # all opened
        net = deepcopy(self.wn)
        self.set_all_elements(net, 'Open')
        metrics, total_demand, total_supply = evaluate(self.wn)
        norm = sum(utils.normalize_obj(self.benchmark, metrics).values())
        self.record_results('all_open', metrics, norm, total_demand, total_supply)

        self.export()

    def add_control(self, net, elem, time):
        cond_beginning = controls.TimeOfDayCondition( net, relation='=', threshold=0)
        cond = controls.TimeOfDayCondition(net, relation='=', threshold=7*3600)
        cond_end = controls.TimeOfDayCondition(net, relation='=', threshold=24*3600)
        if time == 'night':
            night_close = controls.Control(cond, controls.ControlAction(elem, 'status', 0))
            night_open = controls.Control(cond_beginning, controls.ControlAction(elem, 'status', 1))
            net.add_control('control_night_open' + str(elem.name), night_open)
            net.add_control('control_night_close' + str(elem.name), night_close)

        elif time == 'morning':
            morning_open = controls.Control(cond, controls.ControlAction(elem, 'status', 1))
            morning_close = controls.Control(cond_end, controls.ControlAction(elem, 'status', 0))
            net.add_control('control_morning_open' + str(elem.name), morning_open)
            net.add_control('control_morning_close' + str(elem.name), morning_close)

        return net

    def export(self):
        results = utils.handle_dict(self.results)
        results = pd.DataFrame(results)
        results.T.to_csv(os.path.join(self.output_dir, self.zone + '_valve_control.csv'))
        # wntr.network.io.write_inpfile(net, os.path.join(self.output_dir, self.zone + results.T.index[-1] + '.inp'))


def evaluate(net):
    evaluator = Evaluator([net])
    metrics = utils.round_dict(evaluator.evaluate_scenario(), 4)
    return metrics, evaluator.get_total_demand(), evaluator.get_total_supply()


def iterate_all_pumps_combs(networks_path: str, export_path):
    """ A function to evaluate pumps' models combinations

    :param networks_path:   path to dir with 6 networks (one for every year)
    :param export_path:     path to export results
    :return:                csv file with objectives value for every pumps combination
    """

    df = pd.read_csv(os.path.join(RESOURCES_DIR, "pump_candidates.csv"))
    df['pump_model'] = df['pump_id'] + '-' + df['model']

    models = []
    groups = df.groupby(by='pump_id')
    for group, data in groups:
        models.append(data['pump_model'].tolist())

    pump_combs = (list(itertools.product(*models)))
    results = pd.DataFrame()
    for i, c in tqdm(enumerate(pump_combs), total=len(pump_combs)):
        indicators = replace_pumps_and_evaluate_solution(networks_path, df.loc[df['pump_model'].isin(c)],
                                                         path=str(i) + '.inp')
        pumps = df.loc[df['pump_model'].isin(c)]
        pumps = dict(zip(pumps["pump_id"], pumps["model"]))
        results = pd.concat([results, pd.DataFrame.from_dict({**indicators, **pumps}, orient='index').T], axis=0)
        results.to_csv(export_path)


def replace_pumps_and_evaluate_solution(networks_path, pumps_to_replace: pd.DataFrame, path):
    networks = []
    for file_path in glob.glob(os.path.join(networks_path, '*.inp')):
        net = wntr.network.WaterNetworkModel(file_path)
        for i, row in pumps_to_replace.iterrows():
            net = utils.replace_pumps(net, row["pump_id"], row["model"], row["psv_diameter"], row["psv_setting"])

        networks.append(net)
    sc = Evaluator(networks)
    indicators = sc.evaluate_scenario()

    return indicators


if __name__ == '__main__':
    inp_file_path = os.path.join(RESOURCES_DIR, 'networks', 'Greedy_output', 'y1-no-controls.inp')
    elements_file_path = os.path.join(RESOURCES_DIR, 'valves.json')
    output_path = os.path.join(BASE_DIR, 'output', 'controls', 'test')
    with open(elements_file_path) as f:
        grouped_elements = json.load(f)

        grouped_elements = {'class1': grouped_elements['class1']}
        for group, elements in grouped_elements.items():
            cc = ControlChecker(inp_file_path, group, elements, output_path)
            cc.evaluate_controls()
