from agents.actor_critic_agents.SAC_Discrete import SAC_Discrete
import numpy as np
import time
import torch
from utilities.Utility_Functions import create_actor_distribution
import random


class SAC_Discrete_Safe_Shielding(SAC_Discrete):
    def __init__(self, config):
        SAC_Discrete.__init__(self, config=config)
        self.agent_name = "SAC_Discrete_Safe"
        self.unsafe_transitions = np.load(config.unsafe_path, allow_pickle=True)

    def run_n_episodes(self, num_episodes=None, show_whether_achieved_goal=True, save_and_print_results=True):
        """Runs game to completion n times and then summarises results and saves model (if asked to)"""
        if num_episodes is None: num_episodes = self.config.num_episodes_to_run
        start = time.time()
        while self.episode_number < num_episodes:
            self.reset_game()
            self.step()
            if save_and_print_results: self.save_and_print_result()
        time_taken = time.time() - start
        if show_whether_achieved_goal: self.show_whether_achieved_goal()
        if self.config.save_model: self.locally_save_policy()
        return self.game_full_episode_scores, self.rolling_results, time_taken

    def produce_action_and_action_info(self, state):
        """Given the state, produces an action, the probability of the action, the log probability of the action, and
        the argmax action"""
        original_state = torch.clone(state).squeeze(0).numpy()
        if (len(state.shape) == 4):
            state = torch.squeeze(state, 0)
            state = torch.squeeze(state, 0)
            state = torch.flatten(state)
            state = torch.unsqueeze(state, 0)
        elif len(state.shape) == 3:
            state = torch.flatten(state, 1)

        action_probabilities = self.actor_local(state)
        max_probability_action = torch.argmax(action_probabilities, dim=-1)
        for index, (max_action, origin_state) in enumerate(zip(max_probability_action, original_state)):
            safe = -1
            for transition in self.unsafe_transitions:
                s = transition['state'][0]
                try:
                    max_action = max_action.item()
                except:
                    if (s == origin_state).all() and transition['action'] == max_action:
                        safe = index
                        break

            if safe > -1:
                possible_actions = [0, 1, 2, 3]
                possible_actions.pop(max_action)
                max_probability_action[safe] = random.choice(possible_actions)
        # max_probability_action = torch.argmax(action_probabilities, dim=-1)
        action_distribution = create_actor_distribution(self.action_types, action_probabilities, self.action_size)
        action = action_distribution.sample().cpu()
        # Have to deal with situation of 0.0 probabilities because we can't do log 0
        z = action_probabilities == 0.0
        z = z.float() * 1e-8
        log_action_probabilities = torch.log(action_probabilities + z)
        return max_probability_action, (action_probabilities, log_action_probabilities), max_probability_action
