class DecisionSteps:
    def __init__(self, obs, rewards, agent_ids):
        self.obs = obs
        self.rewards = rewards
        self.agent_ids = agent_ids

    def __len__(self):
        return len(self.agent_ids)
