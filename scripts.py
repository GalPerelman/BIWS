import os
import json
import time
import pandas as pd

import preprocess
import metrics
from greedy import Greedy
from exhaustive import ControlChecker
import utils

pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.set_option("display.precision", 2)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESOURCES_DIR = os.path.join(BASE_DIR, 'resources')
OUTPUT_DIR = os.path.join(BASE_DIR, "output")


if __name__ == "__main__":
    """ Run 1st year greedy """
    # The first year budget is only 500000 because of pumps replacements
    # output_path = os.path.join(OUTPUT_DIR, '2_fcv', '1_greedy_output', time.strftime("%Y%m%d%H%M%S") + '_y1')
    # inp = os.path.join(RESOURCES_DIR, "networks", "2_Input-FCV",  "y1.inp")
    # greedy = Greedy(inp, output_dir=output_path, budget=500000, actions_ratio=0.3, hgl_threshold=0.003, n_leaks=1000,
    #                 reevaluate_ratio=0.03, total_run_time=24, hours_duration=168)
    # greedy.pipes.to_csv(os.path.join(output_path, 'pipes.csv'))
    # greedy.leaks.to_csv(os.path.join(output_path, 'leaks.csv'))
    # greedy.start()

    """ Prepare network for year 2 - change leaks emitter coefficients """
    # Grab the final inp from previous year
    # previous_year_file = os.path.join(OUTPUT_DIR, '2_fcv', '1_greedy_output', '20230606080239_y1', 'y1-finalized.inp')
    # preprocess.change_leaks_coef(previous_year_file, 2, os.path.join(RESOURCES_DIR, 'networks', '2_input-FCV', 'y2.inp'))

    """ Run 2nd year greedy """
    # output_path = os.path.join(OUTPUT_DIR, '2_fcv', '1_greedy_output', time.strftime("%Y%m%d%H%M%S") + '_y2')
    # inp = os.path.join(RESOURCES_DIR, "networks", "2_Input-FCV", "y2.inp")
    # greedy = Greedy(inp, output_dir=output_path, budget=670000,
    #                 actions_ratio=0.3, hgl_threshold=0.003, n_leaks=900, reevaluate_ratio=0.03,
    #                 total_run_time=24, hours_duration=168)
    # greedy.pipes.to_csv(os.path.join(output_path, 'pipes.csv'))
    # greedy.leaks.to_csv(os.path.join(output_path, 'leaks.csv'))
    # greedy.start()

    """ Prepare network for year 3 - change leaks emitter coefficients """
    # Grab the final inp from previous year
    # previous_year_file = os.path.join(OUTPUT_DIR, '2_fcv', '1_greedy_output', '20230608092158_y2', 'y2-finalized.inp')
    # preprocess.change_leaks_coef(previous_year_file, 3, os.path.join(RESOURCES_DIR, 'networks', '2_input-FCV', 'y3.inp'))

    """ Run 3rd year greedy """
    # output_path = os.path.join(OUTPUT_DIR, '2_fcv', '1_greedy_output', time.strftime("%Y%m%d%H%M%S") + '_y3')
    # inp = os.path.join(RESOURCES_DIR, "networks", "2_Input-FCV", "y3.inp")
    # greedy = Greedy(inp, output_dir=output_path, budget=670000,
    #                 actions_ratio=0.3, hgl_threshold=0.003, n_leaks=900, reevaluate_ratio=0.03,
    #                 total_run_time=24, hours_duration=168)
    # greedy.pipes.to_csv(os.path.join(output_path, 'pipes.csv'))
    # greedy.leaks.to_csv(os.path.join(output_path, 'leaks.csv'))
    # greedy.start()

    """ Prepare network for year 4 - change leaks emitter coefficients """
    # Grab the final inp from previous year
    # previous_year_file = os.path.join(OUTPUT_DIR, '2_fcv', '1_greedy_output', '20230610160100_y3', 'y3-finalized.inp')
    # preprocess.change_leaks_coef(previous_year_file, 4, os.path.join(RESOURCES_DIR, 'networks', '2_input-FCV', 'y4.inp'))

    """ Run 4th year greedy """
    # output_path = os.path.join(OUTPUT_DIR, '2_fcv', '1_greedy_output', time.strftime("%Y%m%d%H%M%S") + '_y4')
    # inp = os.path.join(RESOURCES_DIR, "networks", "2_Input-FCV", "y4.inp")
    # greedy = Greedy(inp, output_dir=output_path, budget=670000,
    #                 actions_ratio=0.3, hgl_threshold=0.003, n_leaks=900, reevaluate_ratio=0.03,
    #                 total_run_time=24, hours_duration=168)
    # greedy.pipes.to_csv(os.path.join(output_path, 'pipes.csv'))
    # greedy.leaks.to_csv(os.path.join(output_path, 'leaks.csv'))
    # greedy.start()

    """ Prepare network for year 5 - change leaks emitter coefficients """
    # Grab the final inp from previous year
    # previous_year_file = os.path.join(OUTPUT_DIR, '2_fcv', '1_greedy_output', '20230611104545_y4', 'y4-finalized.inp')
    # preprocess.change_leaks_coef(previous_year_file, 5, os.path.join(RESOURCES_DIR, 'networks', '2_input-FCV', 'y5.inp'))

    """ Run 5th year greedy """
    # output_path = os.path.join(OUTPUT_DIR, '2_fcv', '1_greedy_output', time.strftime("%Y%m%d%H%M%S") + '_y5')
    # inp = os.path.join(RESOURCES_DIR, "networks", "2_Input-FCV", "y5.inp")
    # greedy = Greedy(inp, output_dir=output_path, budget=670000,
    #                 actions_ratio=0.3, hgl_threshold=0.003, n_leaks=900, reevaluate_ratio=0.03,
    #                 total_run_time=24, hours_duration=168)
    # greedy.pipes.to_csv(os.path.join(output_path, 'pipes.csv'))
    # greedy.leaks.to_csv(os.path.join(output_path, 'leaks.csv'))
    # greedy.start()

    """ compare soultions """
    comp_solution = "G:/My Drive/3_Academy/P.hd/2_Research/Projects/BIWS-Comp/Code/postprocess/v2/greedy/finalization"
    score = utils.round_dict(metrics.evaluate_scenario(comp_solution), 3)
    print(f"Battle score: {score}")
    print('===============================================================================================')

    paper_solution = "C:/Users/User/Documents/GitHub/BIWS-Paper/output/fcv/2_final_networks"
    score = utils.round_dict(metrics.evaluate_scenario(paper_solution), 3)
    print(f"Paper score: {score}")
    print('===============================================================================================')

    # solution_path = "output/1_fcv/4_final_networks_controls/y0.inp"
    # score = metrics.evaluate_single_net(solution_path)
    # score = utils.round_dict(score, 3)
    # print(score)
    #
    # solution_path = "output/2_fcv/2_final_networks/y0.inp"
    # score = metrics.evaluate_single_net(solution_path)
    # score = utils.round_dict(score, 3)
    # print(score)
