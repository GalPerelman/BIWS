import os
import pandas as pd
import numpy as np
from typing import List
from copy import deepcopy
import glob
import wntr
import warnings

import utils

warnings.filterwarnings(action='ignore', module='wntr')
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.set_option("display.precision", 8)
np.set_printoptions(precision=8, suppress=True)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESOURCES_DIR = os.path.join(BASE_DIR, 'resources')


class Parameters:
    p0 = 0
    pf = 10
    pref = 20
    e = 0.5


class Evaluator:
    """ Evaluate BIWS indicators for a given set of networks or a single network
        NOTE THAT WNTR CONVERTS ALL UNITS TO SI
    """

    def __init__(self, networks: List[wntr.network.WaterNetworkModel]):
        self.networks_by_year = {i: deepcopy(net) for i, net in enumerate(networks)}
        self.input_validation()
        self.sim_duration_hr = int(self.networks_by_year[0].options.time.duration / 3600)
        for year, net in self.networks_by_year.items():
            net.options.time.duration -= 3600  # WNTR runs t+1 hours

        self.num_years = len(self.networks_by_year)

        self.results_by_year = {}
        self.hours_index_by_year = {}

        self.dem_nodes = self.get_demand_nodes()
        self.num_dem_nodes = len(self.dem_nodes)
        self.expected_demand, self.total_expected_demand = self.get_expected_demand()
        self.leak_nodes = self.get_leak_nodes()

        self.pumps_mfr = pd.read_csv(os.path.join(RESOURCES_DIR, 'pumps_mfr.csv'), index_col=0)  # liter/sec
        self.pipes_data = pd.read_csv(os.path.join(RESOURCES_DIR, 'pipes_data.csv'))
        self.pump_operation_penalty = 10

        self.results = pd.DataFrame()
        self.results_demand = pd.DataFrame()
        self.results_pressure = pd.DataFrame()
        self.results_flow = pd.DataFrame()
        self.results_pump_flow = pd.DataFrame()
        self.results_headloss = pd.DataFrame()
        self.results_speed = pd.DataFrame()
        self.indicators = {}

    def input_validation(self):
        durations = []
        for year, net in self.networks_by_year.items():
            durations.append(net.options.time.duration)
            if net.options.hydraulic.demand_model not in ['PDD', 'PDA']:
                raise Exception('SCENARIO HYDRAULIC NOT PRESSURE DRIVEN')

        if durations.count(durations[0]) != len(durations):
            raise Exception('NOT ALL NETWORKS HAVE SAME DURATION')

    def get_expected_demand(self):
        """ Base demand and patterns does change throughout the study period 3.7 in the instructions file
            Expected demand is calculated according to the first network - year 0
            The total expected demand is for all networks (all years)
        """
        base_dem = self.networks_by_year[0].query_node_attribute('base_demand').loc[self.dem_nodes.index]
        dem_pat = self.networks_by_year[0].get_pattern('P0')
        dem_pat = np.hstack([dem_pat.multipliers[: self.sim_duration_hr]] * self.num_years)
        expected_demand = np.dot(dem_pat[:, None], np.array(base_dem)[None, :])
        expected_demand = pd.DataFrame(data=expected_demand, columns=[base_dem.index])
        total_expected_demand = expected_demand.sum().sum()
        return expected_demand, total_expected_demand

    def clear_outliers(self, col):
        m = col > col.quantile(0.95)
        col.loc[m] = col.loc[~m].mean()
        return col

    def run_hyd(self):
        for year, net in self.networks_by_year.items():
            sim = wntr.sim.EpanetSimulator(net)
            results = sim.run_sim()
            self.results_by_year[year] = results
            time_index = (results.node['demand'].index + year * (3600 + results.node['demand'].index.max())) / 3600
            self.hours_index_by_year[year] = time_index

            dem = results.node['demand']
            dem = dem.apply(lambda x: self.clear_outliers(x), axis=0)
            dem.index = time_index
            self.results_demand = pd.concat([self.results_demand, dem])

            pressure = results.node['pressure']
            pressure.index = time_index
            self.results_pressure = pd.concat([self.results_pressure, pressure])

            flow = results.link['flowrate']
            flow.index = time_index
            self.results_flow = pd.concat([self.results_flow, flow])

            pump_flow = results.link['flowrate'].loc[:, net.pump_name_list]
            pump_flow.index = time_index
            self.results_pump_flow = pd.concat([self.results_pump_flow, pump_flow])

            headloss = results.link['headloss']
            headloss.index = time_index
            self.results_headloss = pd.concat([self.results_headloss, headloss])

            speed = results.link['setting']
            speed.index = time_index
            self.results_speed = pd.concat([self.results_speed, speed])
        utils.remove_files('temp.bin', 'temp.inp', 'temp.rpt')

    def get_demand_nodes(self):
        return self.networks_by_year[0].query_node_attribute('base_demand', np.greater, 0)

    def get_leak_nodes(self):
        return self.networks_by_year[0].query_node_attribute('emitter_coefficient').dropna()

    def filter_supply_outliers(self, col):
        mask = col < -0.5
        col.loc[mask] = col.loc[~mask].mean()
        return col

    def filter_outliers(self, col, upper_lim=np.inf,  lower_lim=-np.inf):
        mask = (col > upper_lim) | (col < lower_lim)
        col.loc[mask] = col.loc[~mask].mean()
        return col

    def get_total_demand(self):
        demand = self.results_demand[self.dem_nodes.index]
        demand = np.where(demand < 0, 0, demand)
        return demand.sum(axis=1).sum()

    def get_total_supply(self, year=False):
        supply = self.results_demand.loc[:, self.networks_by_year[0].reservoir_name_list]
        supply = supply.apply(lambda x: self.filter_supply_outliers(x), axis=0)
        if year:
            supply = supply.loc[self.hours_index_by_year[year]]
        total_supply = supply.sum(axis=0).sum() * -1
        return total_supply

    def get_total_leaks(self, year=False):
        leaks_volume = self.results_demand.loc[:, self.leak_nodes.index]
        leaks_volume = pd.DataFrame(np.where(leaks_volume < 0, 0, leaks_volume), index=leaks_volume.index, columns=leaks_volume.columns)
        if year:
            leaks_volume = leaks_volume.loc[self.hours_index_by_year[year]]
        total_leaks_volume = leaks_volume.sum(axis=0).sum()
        return total_leaks_volume

    def get_tanks_total_volume(self, time_hours: int):
        tanks_list = self.networks_by_year[0].tank_name_list
        tank_res = self.results_pressure.loc[time_hours, tanks_list]
        tanks_pressure = pd.DataFrame(index=tank_res.index, data={'pressure': tank_res.values})
        tanks_pressure['volume'] = tanks_pressure.apply(
            lambda x: x['pressure'] * np.pi * self.networks_by_year[0].get_node(x.name).diameter ** 2 / 4, axis=1)
        return tanks_pressure['volume'].sum()

    def get_wntr_expected_demand(self):
        """ NOT IN USE - Long run times
            See get_expected_demand function instead
        """
        return wntr.metrics.expected_demand(self.wn)[self.dem_nodes.index].sum(axis=1)

    def i1(self):
        pressure_at_dem_nodes = self.results_pressure[self.dem_nodes.index]
        total_service_pressure_hours = pressure_at_dem_nodes.gt(Parameters.p0).sum().sum()
        i1 = total_service_pressure_hours / (self.num_dem_nodes * self.sim_duration_hr * self.num_years)
        return i1

    def i2(self):
        total_continuous_pressure_nodes = 0
        for year in range(self.num_years):
            pressure_at_dem_nodes = self.results_by_year[year].node['pressure'][self.dem_nodes.index]
            total_continuous_pressure_nodes += (pressure_at_dem_nodes > Parameters.p0).all().sum()
        i2 = total_continuous_pressure_nodes / (self.num_dem_nodes * self.num_years)
        return i2

    def i3(self):
        total_leak = self.get_total_leaks()
        total_supply = self.get_total_supply()
        return total_leak / total_supply

    def i4(self):
        demand = self.results_demand[self.dem_nodes.index]
        demand = np.where(demand < 0, 0, demand)
        total_actual_demand = demand.sum(axis=1).sum()
        i4 = total_actual_demand / self.total_expected_demand
        return i4

    def i5(self):
        pressure_dem_nodes = self.results_pressure[self.dem_nodes.index]
        pressure_dem_nodes = np.where(pressure_dem_nodes >= Parameters.pref, Parameters.pref, pressure_dem_nodes)
        pressure_dem_nodes[pressure_dem_nodes < 0] = 0
        i5 = pressure_dem_nodes.sum(axis=0).sum() / (self.sim_duration_hr * self.num_dem_nodes *
                                                     Parameters.pref * self.num_years)
        return i5

    def i6(self):
        total_continuous_pressure_nodes = 0
        for year in range(self.num_years):
            pressure_at_dem_nodes = self.results_by_year[year].node['pressure'][self.dem_nodes.index]
            total_continuous_pressure_nodes += (pressure_at_dem_nodes > Parameters.pf).all().sum()
        i6 = total_continuous_pressure_nodes / (self.num_dem_nodes * self.num_years)
        return i6

    def linear_interpolation_i7(self, row):
        temp = pd.DataFrame(data={'start': row['head_ts_start'], 'end': row['head_ts_end']})
        neg = temp[['start', 'end']].min(axis=1)
        pos = temp[['start', 'end']].max(axis=1)
        temp['interpolate'] = (- neg * row['Length']) / (pos - neg)
        temp.loc[(temp['start'] > 0) & (temp['end'] > 0), 'interpolate'] = 0
        temp.loc[(temp['start'] < 0) & (temp['end'] < 0), 'interpolate'] = 1 * row['Length']
        return temp['interpolate'].max()

    def vectorize_interpolatoion(self, pipes_headloss: pd.DataFrame):
        # initiate two dataframes one for start and one for end nodes pressures
        start_p = pd.DataFrame(pipes_headloss['head_ts_start'].to_list(), index=pipes_headloss.index).T
        end_p = pd.DataFrame(pipes_headloss['head_ts_end'].to_list(), index=pipes_headloss.index).T

        # converting the dataframes to high and low pressure in both edges of the pipe
        high = pd.concat([start_p, end_p]).groupby(level=0).max()
        low = pd.concat([start_p, end_p]).groupby(level=0).min()

        # linear interpolation between low and high
        df = -low / (high - low)

        # handling the cases where both edges are positive or negative
        df = np.where((start_p > 0) & (end_p > 0), 0, df)
        df = np.where((start_p < 0) & (end_p < 0), 1, df)

        # multiply by the pipe length
        df = df @ np.eye(len(pipes_headloss)) * pipes_headloss['Length'].values
        max_negative_length = df.max(axis=0)
        return max_negative_length.sum()

    def i7(self):
        i7 = 0
        for year in range(self.num_years):
            df = self.pipes_headloss_summary(year)
            i7 += self.vectorize_interpolatoion(df)
        i7 /= self.num_years
        return i7

    def get_pump_eff_for_i8(self, col, year):
        """ calculate the efficiency according to formula given in the battle instructions section 3.5-(2)
            Assuming pump curve is given by a single point
            Replaced pumps are tagged in the inp as NEW_PUMP
        """
        net = self.networks_by_year[year]
        pump = net.get_link(col.name)

        pump_speed = self.results_speed.loc[self.hours_index_by_year[year], col.name]

        tag = pump.tag
        bep_flow, bep_head = pump.get_pump_curve().points[0]
        best_eff = 0.8 if tag == 'NEW_PUMP' else 0.65
        eff = best_eff * (2 * ((col / pump_speed) / bep_flow) - ((col / pump_speed) / bep_flow) ** 2)
        return eff

    def i8(self):
        i8 = 0
        for year in range(self.num_years):
            pumps_flows = self.results_pump_flow.loc[self.hours_index_by_year[year]]
            pumps_head = self.results_headloss.loc[self.hours_index_by_year[year], self.results_pump_flow.columns] * -1

            pumps_eff = pumps_flows.apply(lambda x: self.get_pump_eff_for_i8(x, year))
            pumps_power_kwatt = (9810 * pumps_flows * pumps_head) / (1000 * pumps_eff)
            pumps_power_kwatt = pumps_power_kwatt.apply(lambda x: self.filter_outliers(x, upper_lim=50, lower_lim=0), axis=0)
            i8 += pumps_power_kwatt.sum(axis=0).sum()
        return i8

    def i9(self):
        sr = pd.Series(dtype=float)
        for year in range(self.num_years):
            demand = self.results_by_year[year].node['demand'][self.dem_nodes.index]
            demand = pd.DataFrame(np.where(demand < 0, 0, demand), index=demand.index, columns=demand.columns)
            actual_demand = demand.sum(axis=0)

            expected = self.expected_demand.iloc[:len(actual_demand)].sum(axis=0)
            expected.index = actual_demand.index
            sr = pd.concat([sr, actual_demand / expected])

        sr[sr > 1] = 1  # Due to precision errors supply can get larger values than the demand
        asr = sr.sum() / (self.num_dem_nodes * self.num_years)
        adev = np.abs(sr - asr).sum(axis=0).sum() / (self.num_dem_nodes * self.num_years)
        i9 = 1 - adev / asr
        return i9

    def evaluate_scenario(self):
        self.run_hyd()
        self.indicators[1] = self.i1()
        self.indicators[2] = self.i2()
        self.indicators[3] = self.i3()
        self.indicators[4] = self.i4()
        self.indicators[5] = self.i5()
        self.indicators[6] = self.i6()
        self.indicators[7] = self.i7()
        self.indicators[8] = self.i8()
        self.indicators[9] = self.i9()
        return self.indicators

    def norm_min_obj(self, benchmark_obj):
        obj = self.evaluate_scenario()
        df = pd.DataFrame.from_dict(obj, orient='index', columns=['x'])
        df = pd.merge(df, benchmark_obj, left_index=True, right_index=True)
        df['normalized'] = (df['x'] - df['best']) / (df['worse'] - df['best'])
        df['normalized'] = np.where(df['normalized'] < 0, 0, df['normalized'])
        df['normalized'] = np.where(df['normalized'] > 1, 1, df['normalized'])
        return df['normalized'].to_dict(), obj

    def single_norm_min_obj(self, benchmark_obj=False):
        normalized, obj = self.norm_min_obj(benchmark_obj)
        return sum(list(normalized.values())), obj

    def pump_flow_penalty(self):
        df = self.results_pump_flow.copy() * 1000  # 1000 is for converting CM/s to l/s
        df = df[list(set(self.pumps_mfr.index.tolist()) & set(df.columns))]
        for col in df.columns:
            df.loc[:, col] = np.where(df[col] > self.pumps_mfr.loc[col, 'mfr'], 1, 0)
        count_above_mfr = df.sum(axis=0).sum()
        return count_above_mfr * self.pump_operation_penalty

    def leaks_summary(self):
        df1 = self.results_pressure[self.results_pressure.columns.intersection(self.leak_nodes.index)].mean(axis=0)
        df2 = self.results_demand[self.results_demand.columns.intersection(self.leak_nodes.index)].sum(axis=0) * 1000
        df2[df2 < 0] = 0

        df = pd.concat([df1, df2], axis=1)
        df.columns = ['avg_pressure', 'total_water_loss']

        leaks_data = pd.read_csv(os.path.join(RESOURCES_DIR, "preprocess", "leaks_preprocess.csv"),
                                 usecols=['Link', 'Link_id', 'Coef', 'Leak_id'])
        df = pd.merge(df, leaks_data, left_index=True, right_on="Leak_id")
        df = pd.merge(df, self.pipes_data[['ID', 'Diameter']], left_on='Link', right_on='ID')

        df['detect_cost'] = 2400 * np.exp(-28 * df['Coef'])
        df['repair_cost'] = (94 - 0.3 * df['Diameter'] + 0.01 * df['Diameter'] ** 2) * (
                    1.5 + 0.11 * np.log10(df['Coef']))
        df['total_cost'] = df['detect_cost'] + df['repair_cost']
        return df

    def pipes_headloss_summary(self, year):
        df_head = self.results_by_year[year].node['head'].T

        head_ts = df_head.apply(lambda x: np.array(x), axis=1)
        head_ts.name = 'head_ts'

        df = pd.merge(self.pipes_data, head_ts, left_on='Node1', right_index=True)
        df = pd.merge(df, head_ts, left_on='Node2', right_index=True, suffixes=('_start', '_end'))
        df['headloss_ts'] = df['head_ts_end'] - df['head_ts_start']
        df.loc[:, 'mean_head_loss'] = df.apply(lambda x: np.mean(x['headloss_ts']), axis=1)
        df.loc[:, 'min_head_loss'] = df.apply(lambda x: np.min(x['headloss_ts']), axis=1)
        df.loc[:, 'max_head_loss'] = df.apply(lambda x: np.max(x['headloss_ts']), axis=1)
        df = df.sort_values('max_head_loss', ascending=False)
        return df


def evaluate_scenario(path_to_networks):
    """ evaluate a complete scenario - 6 networks
        path_to_networks - path to a directory with 6 networks
        networks names should be y<i>.inp such that <i> represent the year
    """
    import warnings
    warnings.filterwarnings('ignore')

    networks = []
    for file_path in glob.glob(os.path.join(path_to_networks, '*.inp')):
        net = wntr.network.WaterNetworkModel(file_path)
        networks.append(net)

    indicators = Evaluator(networks).evaluate_scenario()
    return indicators


if __name__ == "__main__":
    # Usage example
    inp_path = os.path.join(RESOURCES_DIR, 'networks', 'BIWS_y0.inp')
    wn = wntr.network.WaterNetworkModel(inp_path)
    # print(Evaluator([wn]).evaluate_scenario())


    wn.options.time.duration = 5 * 3600
    evaluator = Evaluator([wn])
    evaluator.run_hyd()



