import numpy as np


class Trajectory:
    def __init__(self, gamma=0.99):
        self.obs = []
        self.move_actions = []
        self.jump_actions = []
        self.rewards = []
        self.log_probs = []
        self.state_vals = []
        self.gamma = gamma
        self.is_interrupted = False
        self.is_terminal = False

        self.len = 0

    def insert(
        self,
        obs,
        move_action,
        jump_action,
        reward,
        log_prob,
        state_val,
    ):
        self.obs.append(obs)
        self.move_actions.append(move_action)
        self.jump_actions.append(jump_action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        self.state_vals.append(state_val)
        self.len += 1

    def set_is_terminal(self):
        self.is_terminal = True

    def set_is_interrupted(self):
        self.is_interrupted = True

    def get_reward_to_go(self):
        rewards = np.array(self.rewards, dtype=np.float32)
        returns = np.zeros_like(rewards, dtype=np.float32)

        curr_reward = self.state_vals[-1] if self.is_interrupted else 0.0

        for t in reversed(range(len(rewards))):
            curr_reward = rewards[t] + (self.gamma * curr_reward)
            returns[t] = curr_reward

        return returns

    def get_advantages(self, rtgs):
        state_vals = np.array(self.state_vals, dtype=np.float32)

        advantages = rtgs - state_vals
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return advantages

    def __len__(self):
        return self.len


if __name__ == "__main__":
    traj = Trajectory()
    # traj.insert(5, 5, 3, 2)
    # traj.insert(5, 5, 3, 2)
    # traj.insert(5, 5, 3, 2)
    # traj.insert(5, 5, 3, 2)

    # print(traj.get_advantages())
