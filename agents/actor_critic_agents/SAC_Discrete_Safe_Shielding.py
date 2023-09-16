from agents.actor_critic_agents.SAC_Discrete import SAC_Discrete
import numpy as np
import time
import torch
from utilities.Utility_Functions import create_actor_distribution

class SAC_Discrete_Safe(SAC_Discrete):
    def __init__(self, config):
        SAC_Discrete.__init__(self, config=config)
        self.agent_name = "SAC_Discrete_Safe"
        self.unsafe_transitions = np.load(config.unsafe_path, allow_pickle=True)

    def save_experience(self, memory=None, experience=None):
        """Saves the recent experience to the memory buffer"""
        if memory is None: memory = self.memory
        if experience is None: experience = self.state, self.action, self.reward, self.next_state, self.done
        # do not add the experience to the replay buffer if it's an unsafe action.
        # safe = True
        # for _, transition in self.unsafe_transition.iterrows():
        #     if transition['state'] == self.state and transition['action'] == self.action:
        #         safe = False
        #         break
        #
        # if safe:
        #     memory.add_experience(*experience)
        safe = True
        # for transition in self.unsafe_transitions:
        #     if (transition['state'] == self.state).all() and transition['action'] == self.action:
        #         print('UNSAFE TRANSITION DETECTED')
        #         safe = False
        #         break
        if safe:
            memory.add_experience(*experience)
    
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
                if action_probabilities.shape[0] == 256:
                    pass
                new_prob = np.array([0.3333, 0.3333, 0.3333, 0.3333])
                new_prob[max_action] = 0.0001
                action_probabilities[safe] = torch.tensor(new_prob)

        max_probability_action = torch.argmax(action_probabilities, dim=-1)
        action_distribution = create_actor_distribution(self.action_types, action_probabilities, self.action_size)
        action = action_distribution.sample().cpu()
        # Have to deal with situation of 0.0 probabilities because we can't do log 0
        z = action_probabilities == 0.0
        z = z.float() * 1e-8
        log_action_probabilities = torch.log(action_probabilities + z)
        return action, (action_probabilities, log_action_probabilities), max_probability_action
