from py_experimenter.experimenter import PyExperimenter
from experiment_execution import run_ex
import os
import pandas as pd


def initialize_experiments(experimenter):

    file_path = os.path.split(__file__)[0]+"/datasets.csv"
    df = pd.read_csv(file_path)
    eval_func = ['kfold_5', 'lccv-80', 'mccv_5', "lce", "pfn"]
    dataset_id = []
    for id in df["openmlid"]:
        dataset_id.append(int(id))
    experimenter.fill_table_from_combination(
        parameters={
            "dataset_id": dataset_id,
            "eval_func": eval_func
        }
    )


if __name__ == '__main__':

    configuration_path = os.path.split(__file__)[0]+'/configuration.conf'
    experimenter = PyExperimenter(
        experiment_configuration_file_path=configuration_path,
        name='example',
        use_codecarbon=False
    )
    initialize_experiments(experimenter=experimenter)
