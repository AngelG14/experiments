from experiment_execution import run_ex
import logging


def initialize_loggers(naiveautoml_level=None, lccv_level=None):

    logger_levels = {
        "DEBUG": logging.DEBUG,
        "WARM": logging.WARN,
        "INFO": logging.INFO
    }

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    if naiveautoml_level is not None:

        # NaiveAutoML Logger
        logger = logging.getLogger('naiveautoml')
        logger.setLevel(logger_levels[naiveautoml_level])

        # File Handler
        naiveautoml_file_handler = logging.FileHandler('naiveautoml.log')
        naiveautoml_file_handler.setLevel(logger_levels[naiveautoml_level])
        naiveautoml_file_handler.setFormatter(formatter)
        logger.addHandler(naiveautoml_file_handler)

    if lccv_level is not None:
        # LCCV Logger
        lccv_logger = logging.getLogger("lccv")
        lccv_logger.setLevel(logger_levels[lccv_level])
        elm_logger = logging.getLogger("elm")
        elm_logger.setLevel(logger_levels[lccv_level])

        # File handler
        file_handler2 = logging.FileHandler('lccv.log')
        file_handler2.setLevel(logger_levels[lccv_level])
        file_handler2.setFormatter(formatter)
        lccv_logger.addHandler(file_handler2)
        elm_logger.addHandler(file_handler2)

if __name__ == '__main__':

    configuration_path = os.path.split(__file__)[0]+'/configuration.conf'
    experimenter = PyExperimenter(
        experiment_configuration_file_path=configuration_path,
        name='example',
        use_codecarbon=False
    )
    initialize_experiments(experimenter=experimenter)