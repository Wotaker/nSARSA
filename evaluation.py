from __future__ import annotations

import numpy as np
import sklearn.preprocessing as skl_preprocessing

from problem import Corner, Experiment, Environment
from solution import OffPolicyNStepSarsaDriver
import utils


def main() -> None:

    load_id = 'td5000-mapc-n5-a0.3'

    experiment = Experiment(
        environment=Environment(
            corner=Corner(
                name='corner_c'
            ),
            steering_fail_chance=0.01,
        ),
        driver=utils.load(f'drivers/{load_id}.pkl'),
        number_of_episodes=10,
        drawing_frequency=1,
        save_prefix=f'evaluation/{load_id}',
    )
    experiment.evaluate()


if __name__ == '__main__':
    main()
