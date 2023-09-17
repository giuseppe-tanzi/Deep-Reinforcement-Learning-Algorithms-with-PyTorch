from agents.actor_critic_agents.SAC_Discrete import SAC_Discrete
import numpy as np
import time

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
        safe = True
        for transition in self.unsafe_transitions:
            if (transition['state'] == self.state).all() and transition['action'] == self.action:
                print('UNSAFE TRANSITION DETECTED')
                safe = False
                break
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
