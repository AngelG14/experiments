from py_experimenter.experimenter import PyExperimenter
import os
import pandas as pd
from experiment_utils import get_dataset

file_path = os.path.split(__file__)[0]


def initialize_experiments(experimenter):

    database_path = file_path + "/datasets.csv"
    df = pd.read_csv(database_path, sep=";")
    eval_func = ['kfold_5', 'lccv-80', 'mccv_5', "lce", "pfn"]
    dataset_id = []
    for id in df["openmlid"]:
        dataset_id.append(int(id))
        get_dataset(id)
    experimenter.fill_table_from_combination(
        parameters={
            "dataset_id": dataset_id,
            "eval_func": eval_func
        }
    )
    print(experimenter.get_table())


if __name__ == '__main__':

    configuration_path = file_path+'/config/experiment_configuration.cfg'
    experimenter = PyExperimenter(
        experiment_configuration_file_path=configuration_path,
        name='example',
        use_codecarbon=False
    )
    initialize_experiments(experimenter=experimenter)
