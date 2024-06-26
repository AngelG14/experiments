from experiment_execution import run_ex
from py_experimenter.experimenter import PyExperimenter
import logging
import os

file_path = os.path.split(__file__)[0]


def initialize_loggers(naiveautoml_level=None, lccv_level=None):

    logger_levels = {
        "DEBUG": logging.DEBUG,
        "WARM": logging.WARN,
        "INFO": logging.INFO
    }

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    if naiveautoml_level is not None:

        # Console handler
        naiveautoml_ch = logging.StreamHandler()
        naiveautoml_ch.setLevel(logger_levels[naiveautoml_level])
        naiveautoml_ch.setFormatter(formatter)

        # NaiveAutoML Logger
        naiveautoml_logger = logging.getLogger('naiveautoml')
        naiveautoml_logger.setLevel(logger_levels[naiveautoml_level])

        # File Handler
        # naiveautoml_file_handler = logging.FileHandler('naiveautoml.log')
        # naiveautoml_file_handler.setLevel(logger_levels[naiveautoml_level])
        # naiveautoml_file_handler.setFormatter(formatter)
        # naiveautoml_logger.addHandler(naiveautoml_file_handler)

        naiveautoml_logger.addHandler(naiveautoml_ch)

    if lccv_level is not None:

        # Console handler
        lccv_ch = logging.StreamHandler()
        lccv_ch.setLevel(logger_levels[naiveautoml_level])
        lccv_ch.setFormatter(formatter)

        # LCCV Logger
        lccv_logger = logging.getLogger("lccv")
        lccv_logger.setLevel(logger_levels[lccv_level])
        elm_logger = logging.getLogger("elm")
        elm_logger.setLevel(logger_levels[lccv_level])

        # File handler
        # file_handler2 = logging.FileHandler('lccv.log')
        # file_handler2.setLevel(logger_levels[lccv_level])
        # file_handler2.setFormatter(formatter)
        # lccv_logger.addHandler(file_handler2)
        # elm_logger.addHandler(file_handler2)

        lccv_logger.addHandler(lccv_ch)
        elm_logger.addHandler(lccv_ch)


if __name__ == '__main__':

    configuration_path = file_path+'/config/experiment_configuration.cfg'
    experimenter = PyExperimenter(
        experiment_configuration_file_path=configuration_path,
        name='example',
        use_codecarbon=False
    )
    initialize_loggers(naiveautoml_level="INFO", lccv_level="INFO")
    experimenter.execute(
        experiment_function=run_ex,
        random_order=False
    )
