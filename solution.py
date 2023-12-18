from __future__ import annotations
import collections
import random

import numpy as np
import sklearn.preprocessing as skl_preprocessing
import argparse

from problem import Action, available_actions, Corner, Driver, Experiment, Environment, State
import utils

ALMOST_INFINITE_STEP = 1e5
MAX_LEARNING_STEPS = 500


class RandomDriver(Driver):
    def __init__(self):
        self.current_step: int = 0

    def start_attempt(self, state: State) -> Action:
        self.current_step = 0
        return random.choice(available_actions(state))

    def control(self, state: State, last_reward: int) -> (Action, bool):
        self.current_step += 1
        return random.choice(available_actions(state)), True

    def finished_learning(self) -> bool:
        return self.current_step > MAX_LEARNING_STEPS


class OffPolicyNStepSarsaDriver(Driver):
    def __init__(self, step_size: float, step_no: int, experiment_rate: float, discount_factor: float) -> None:
        self.step_size: float = step_size
        self.step_no: int = step_no
        self.experiment_rate: float = experiment_rate
        self.discount_factor: float = discount_factor
        self.q: dict[tuple[State, Action], float] = collections.defaultdict(float)
        self.current_step: int = 0
        self.final_step: int = ALMOST_INFINITE_STEP
        self.finished: bool = False
        self.states: dict[int, State] = dict()
        self.actions: dict[int, Action] = dict()
        self.rewards: dict[int, int] = dict()

    def start_attempt(self, state: State) -> Action:
        self.current_step = 0
        self.states[self._access_index(self.current_step)] = state
        action = self._select_action(self.epsilon_greedy_policy(state, available_actions(state)))
        self.actions[self._access_index(self.current_step)] = action
        self.final_step = ALMOST_INFINITE_STEP
        self.finished = False
        return action

    def control(self, state: State, last_reward: int) -> (Action, bool):
        explored = False
        if self.current_step < self.final_step:
            self.rewards[self._access_index(self.current_step + 1)] = last_reward
            self.states[self._access_index(self.current_step + 1)] = state
            if self.final_step == ALMOST_INFINITE_STEP and (
                    last_reward == 0 or self.current_step == MAX_LEARNING_STEPS
            ):
                self.final_step = self.current_step
            action = self._select_action(self.epsilon_greedy_policy(state, available_actions(state)))
            self.actions[self._access_index(self.current_step + 1)] = action
        else:
            action = Action(0, 0)

        update_step = self.current_step - self.step_no + 1
        if update_step >= 0:
            return_value_weight = self._return_value_weight(update_step)
            return_value = self._return_value(update_step)
            state_t = self.states[self._access_index(update_step)]
            action_t = self.actions[self._access_index(update_step)]

            # TODO: Tutaj trzeba zaktualizować tablicę wartościującą akcje Q
            explored = self.q[state_t, action_t] == 0
            self.q[state_t, action_t] += self.step_size * return_value_weight * (return_value - self.q[state_t, action_t])

        if update_step == self.final_step - 1:
            self.finished = True

        self.current_step += 1
        return action, explored
    
    def evaluate(self, state: State, last_reward: int) -> Action:
        self.finished = last_reward == 0
        return self._select_action(self.greedy_policy(state, available_actions(state)))

    def _return_value(self, update_step):
        # TODO (DONE): Tutaj trzeba policzyć zwrot G
        return_value = 0.0
        for i in range(update_step + 1, min(update_step + self.step_no, self.final_step) + 1):
            discount = self.discount_factor ** (i - update_step - 1)
            reward = self.rewards[self._access_index(i)]
            return_value += discount * reward
        
        if update_step + self.step_no < self.final_step:
            discount = self.discount_factor ** self.step_no
            estimate = self.q[self.states[self._access_index(update_step + self.step_no)], self.actions[self._access_index(update_step + self.step_no)]]
            return_value += discount * estimate
        
        return return_value

    def _return_value_weight(self, update_step):
        # TODO (DONE): Tutaj trzeba policzyć korektę na różne prawdopodobieństwa ρ (ponieważ uczymy poza-polityką)
        return_value_weight = 1.0
        for i in range(update_step + 1, min(update_step + self.step_no, self.final_step - 1) + 1):
            state_i = self.states[self._access_index(i)]
            action_i = self.actions[self._access_index(i)]
            pi = self.greedy_policy(state_i, available_actions(state_i))[action_i]
            behavior = self.epsilon_greedy_policy(state_i, available_actions(state_i))[action_i]
            return_value_weight *= pi / behavior
        return return_value_weight

    def finished_learning(self) -> bool:
        return self.finished

    def _access_index(self, index: int) -> int:
        return index % (self.step_no + 1)

    @staticmethod
    def _select_action(actions_distribution: dict[Action, float]) -> Action:
        actions = list(actions_distribution.keys())
        probabilities = list(actions_distribution.values())
        i = np.random.choice(list(range(len(actions))), p=probabilities)
        return actions[i]

    def epsilon_greedy_policy(self, state: State, actions: list[Action]) -> dict[Action, float]:
        # TODO: tutaj trzeba ustalic prawdopodobieństwa wyboru akcji według polityki ε-zachłannej
        values = [self.q[state, action] for action in actions]
        maximal_spots = (values == np.max(values)).astype(float)
        probabilities = maximal_spots * (1 - self.experiment_rate) / (np.sum(maximal_spots))
        probabilities += np.ones_like(values) * self.experiment_rate / len(values)

        # if np.sum(probabilities) != 1.0:
        #     print(f"[EXCEPTION] probabilities sum to {np.sum(probabilities)}: {probabilities}")
        #     raise Exception("probabilities do not sum to 1.0")

        return {action: probability for action, probability in zip(actions, probabilities)}

    def greedy_policy(self, state: State, actions: list[Action]) -> dict[Action, float]:
        probabilities = self._greedy_probabilities(state, actions)
        return {action: probability for action, probability in zip(actions, probabilities)}

    def _greedy_probabilities(self, state: State, actions: list[Action]) -> np.ndarray:
        values = [self.q[state, action] for action in actions]
        maximal_spots = (values == np.max(values)).astype(float)
        probabilities = self._normalise(maximal_spots)
        return probabilities

    @staticmethod
    def _random_probabilities(actions: list[Action]) -> np.ndarray:
        maximal_spots = np.array([1.0 for _ in actions])
        return OffPolicyNStepSarsaDriver._normalise(maximal_spots)

    @staticmethod
    def _normalise(probabilities: np.ndarray) -> np.ndarray:
        # return skl_preprocessing.normalize(probabilities.reshape(1, -1), norm='l1')[0]
        prob_sum = np.sum(probabilities)
        if prob_sum == 0:
            return np.ones_like(probabilities) / len(probabilities)
        return probabilities / prob_sum


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='nSARSA Experiment')
    parser.add_argument('--n_step', '-n', type=int, default=5, help='Number of steps for n-step SARSA')
    parser.add_argument('--alpha', '-a', type=float, default=0.3, help='Step size (learning rate)')
    parser.add_argument('--epsilon', '-e', type=float, default=0.05, help='Exploration rate')
    parser.add_argument('--gamma', '-g', type=float, default=1.0, help='Discount factor')
    parser.add_argument('--no_episodes', '-ne', type=int, default=5000, help='Number of episodes')
    parser.add_argument('--map', '-m', type=str, default='c', help='Map type')
    parser.add_argument('--silent', '-s', action='store_true', help='Silent mode')
    args = parser.parse_args()

    n_step = args.n_step
    alpha = args.alpha
    epsilon = args.epsilon
    gamma = args.gamma
    no_episodes = args.no_episodes
    map = args.map
    silent = args.silent

    id = f'td{no_episodes}-map{map}-n{n_step}-a{alpha}'

    experiment = Experiment(
        environment=Environment(
            corner=Corner(
                name=f'corner_{map}'
            ),
            steering_fail_chance=0.01,
        ),
        driver=OffPolicyNStepSarsaDriver(
            step_no=n_step,
            step_size=alpha,
            experiment_rate=epsilon,
            discount_factor=gamma,
        ),
        number_of_episodes=no_episodes,
        drawing_frequency=int(no_episodes / 10),
        save_prefix=f'plots/{id}',
        silent=silent
    )

    experiment.run()
    experiment.save_driver(f"drivers/{id}.pkl")
    experiment.save_results(id)
