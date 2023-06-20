from agents.actor_critic_agents.SAC_Discrete import SAC_Discrete
import pandas as pd

class SAC_Discrete_Safe(SAC_Discrete):
    def __init__(self, config):
        SAC_Discrete.__init__(config=config)
        self.agent_name = "SAC_Discrete_Safe"
        self.unsafe_transition = pd.read_csv(config.unsafe_path)

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

        if 5 in self.state and 5 not in self.next_state:
            pass
        else:
            memory.add_experience(*experience)
