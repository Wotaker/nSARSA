from __future__ import annotations

import numpy as np
import sklearn.preprocessing as skl_preprocessing

from problem import Corner, Experiment, Environment
from solution import OffPolicyNStepSarsaDriver
import utils


LOAD_ID = 'td20000-mapd-n4-a0.512'
MAP = "d"

def main() -> None:

    experiment = Experiment(
        environment=Environment(
            corner=Corner(
                name=f'corner_{MAP}'
            ),
            steering_fail_chance=0.01,
        ),
        driver=utils.load(f'drivers/{LOAD_ID}.pkl'),
        number_of_episodes=10,
        drawing_frequency=1,
        save_prefix=f'evaluation/{LOAD_ID}',
    )
    experiment.evaluate()


if __name__ == '__main__':
    main()
