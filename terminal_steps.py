class TerminalSteps:
    def __init__(self, obs, rewards, agent_ids, is_interrupted):
        self.obs = obs
        self.rewards = rewards
        self.agent_ids = agent_ids
        self.is_interrupted = is_interrupted

    def __len__(self):
        return len(self.agent_ids)
