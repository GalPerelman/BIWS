import os
import numpy as np
import pandas as pd


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
