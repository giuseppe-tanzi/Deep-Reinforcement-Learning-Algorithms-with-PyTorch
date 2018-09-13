import copy

import torch

from Base_Agent import Base_Agent
from Linear_Model import Linear_Model

import numpy as np



class Hill_Climbing_Agent(Base_Agent):

    def __init__(self, environment, seed, hyperparameters, rolling_score_length, average_score_required,
                 agent_name):

        self.hyperparameters = hyperparameters["Stochastic_Policy_Search_Agents"]

        Base_Agent.__init__(self, environment=environment,
                            seed=seed, hyperparameters=self.hyperparameters, rolling_score_length=rolling_score_length,
                            average_score_required=average_score_required, agent_name=agent_name)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        if self.hyperparameters["policy_network_type"] == "Linear":
            self.policy = Linear_Model(self.state_size, self.action_size)
            self.best_weights_seen = self.policy.weights

        self.best_episode_score_seen = float("-inf")

        self.stochastic_action_decision = self.hyperparameters["stochastic_action_decision"]
        self.noise_scale = self.hyperparameters["noise_scale_start"]
        self.noise_scale_min = self.hyperparameters["noise_scale_min"]
        self.noise_scale_max = self.hyperparameters["noise_scale_max"]
        self.noise_scale_growth_factor = self.hyperparameters["noise_scale_growth_factor"]


    def step(self):
        """Runs a step within a game including a learning step if required"""
        self.pick_and_conduct_action()

        self.update_next_state_reward_done_and_score()

        if self.time_to_learn():
            self.learn()

        self.save_experience()
        self.state = self.next_state #this is to set the state for the next iteration

    def pick_and_conduct_action(self):
        self.action = self.pick_action()
        self.conduct_action()

    def pick_action(self):

        policy_values = self.policy.forward(self.state)

        if self.stochastic_action_decision:
            action = np.random.choice(self.action_size, p=policy_values) # option 1: stochastic policy
        else:
            action = np.argmax(policy_values)  # option 2: deterministic policy

        return action


    def time_to_learn(self):
        """Tells agent to perturb weights at end of every episode"""
        return self.done

    def learn(self):

        raw_noise = (2.0*(np.random.rand(*self.policy.weights.shape) - 0.5))

        if self.score >= self.best_episode_score_seen:

            self.best_episode_score_seen = self.score
            self.best_weights_seen = self.policy.weights
            noise_scale = max(self.noise_scale_min, self.noise_scale / self.noise_scale_growth_factor)
            self.policy.weights += noise_scale * raw_noise

        else:

            noise_scale = min(self.noise_scale_max, self.noise_scale * self.noise_scale_growth_factor)
            self.policy.weights = self.best_weights_seen + noise_scale * raw_noise

    def save_experience(self):
        pass



    def locally_save_policy(self):
        pass
