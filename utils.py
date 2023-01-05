import os
import numpy as np
import pandas as pd
import wntr
import wntr.network.controls as controls
from functools import singledispatch

#  General utils
def validate_dir_path(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def remove_files(*files):
    for file in files:
        if file and os.path.exists(file) and not os.path.isdir(file):
            os.remove(file)


def get_file_name_from_path(path):
    file_name = os.path.basename(path)
    file_name, ext = os.path.splitext(file_name)
    return file_name, ext


def round_dict(d, digits):
    return {k: round(v, digits) for k, v in d.items()}


@singledispatch
def keys_to_strings(ob):
    return ob


@keys_to_strings.register
def handle_dict(ob: dict):
    return {str(k): keys_to_strings(v) for k, v in ob.items()}


# EPANET utils (using wntr)

def set_valve_status(net, valve_name, status):
    """

    :param net:             The net where valve status is changed (wntr.metwork.WaterNetworkModel)
    :param valve_name:      Valve id
    :param status:          Required status (str)
    :return:                Modified net
    """
    net.get_link(valve_name).initial_status = 'Open'
    return net


def add_status_by_time_control(net, element, start, value, name):
    cond = controls.TimeOfDayCondition(net, relation='=', threshold=start * 3600)
    cont = controls.Control(cond, controls.ControlAction(element, 'status', value))
    net.add_control(name + str(element.name), cont)
    return net


# Greedy utils

DIAMETERS = np.array([50, 63, 75, 100, 125, 150, 200, 250, 300, 350, 400, 450, 500, 600, 700, 800])  # mm
DIAMETERS_m = DIAMETERS / 1000


def get_next_diameter(diam_mm):
    return DIAMETERS[DIAMETERS > diam_mm].min()


def replace_pipe_cost(new_diam_m, length):
    return (13 + 29 * new_diam_m + 1200 * new_diam_m ** 2) * length


def fix_leak_cost(coef, diameter_mm):
    """ coef needs to have (l/s) / m units """
    if coef == 0:
        return 0
    else:
        detect_cost = 2400 * np.exp(-28 * coef)
        repair_cost = (94 - 0.3 * diameter_mm + 0.01 * diameter_mm ** 2) * (1.5 + 0.11 * np.log10(coef))
        return detect_cost + repair_cost


def get_benchmarks(metrics: pd.DataFrame):
    """
    Get the best and worst for each metric
    :param metrics:     pd.DataFrame metrics columns are name 1-9
    :return:            two dictionaries of best and worst values
    """
    # Clear invalid results
    # remove rows with negative values
    metrics = metrics[(metrics > 0).all(1)]
    # remove rows with values larger than 1 (except from metric 7, 8)
    metrics = metrics.drop(metrics[(metrics['1'] > 1) | (metrics['2'] > 1) | (metrics['3'] > 1)
                                   | (metrics['4'] > 1) | (metrics['5'] > 1) | (metrics['6'] > 1)
                                   | (metrics['9'] > 1)].index, axis=0)

    mx = metrics.max(axis=0).to_dict()
    mn = metrics.min(axis=0).to_dict()
    best = {1: mx['1'], 2: mx['2'], 3: mn['3'], 4: mx['4'], 5: mx['5'], 6: mx['6'], 7: mn['7'], 8: mn['8'], 9: mx['9']}
    worst = {1: mn['1'], 2: mn['2'], 3: mx['3'], 4: mn['4'], 5: mn['5'], 6: mn['6'], 7: mx['7'], 8: mx['8'], 9: mn['9']}
    return best, worst


def normalize_obj(worse, objectives):
    # best is 1 for max objectives and 0 for min objectives
    best = pd.DataFrame(index=range(1, 10), data=[1, 1, 0, 1, 1, 1, 0, 0, 1], columns=['best'])
    worse = pd.DataFrame.from_dict(worse, orient='index', columns=['worse'])
    objectives = pd.DataFrame.from_dict(objectives, orient='index', columns=['objectives'])

    df = pd.merge(worse, best, left_index=True, right_index=True)
    df = pd.merge(df, objectives, left_index=True, right_index=True)
    df['normalized'] = (df['objectives'] - df['best']) / (df['worse'] - df['best'])
    df['normalized'] = np.where(df['normalized'] < 0, 0, df['normalized'])
    df['normalized'] = np.where(df['normalized'] > 1, 1, df['normalized'])
    df['normalized'] = np.where(df['normalized'].isnull(), df['best'], df['normalized'])
    return df['normalized'].to_dict()


def replace_pumps(net, pump_id: str, model: str, v_diameter, v_setting, v_type='PSV'):
    pump = net.get_link(pump_id)
    pump.pump_curve_name = model  # replace the pump model - changing pump curve

    if pd.isnull(v_diameter) and pd.isnull(v_setting):  # Case 1 - replace old pump without adding valve
        pump.tag = "NEW_PUMP"
        return net

    elif pump.tag == "NEW_PUMP":  # Case 2 - replace new pump and set valve settings
        valve = net.get_link(net.get_links_for_node(pump.end_node_name)[0])
        valve.diameter = v_diameter
        valve.initial_setting = v_setting
        valve.type = v_type

    else:  # Case 3 - replace old pump and add valve
        pump.tag = "NEW_PUMP"
        pump.initial_status = 1
        dis_node = net.get_node(pump.end_node_name)
        dis_node_name = dis_node.name
        net.add_junction(pump_id + '_discharge', elevation=dis_node.elevation, coordinates=dis_node.coordinates)
        pump.end_node = net.get_node(pump_id + '_discharge')

        net.add_valve(pump_id + '_v', start_node_name=pump_id + '_discharge', end_node_name=dis_node_name,
                      diameter=v_diameter, valve_type=v_type, initial_setting=v_setting)

    return net
