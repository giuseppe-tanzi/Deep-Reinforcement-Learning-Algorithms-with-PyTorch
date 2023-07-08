from agents.actor_critic_agents.SAC_Discrete import SAC_Discrete
import numpy as np

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
        for transition in self.unsafe_transitions:
            if (transition['state'] == self.state).all() and transition['action'] == self.action:
                print('UNSAFE TRANSITION DETECTED')
                safe = False
                break
        if safe:
            memory.add_experience(*experience)
