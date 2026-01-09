import torch
from trajectory import Trajectory


class TrajectoryBuffer:
    def __init__(self):
        self.buffer: dict[int, Trajectory] = {}
        self.len = 0

    def add(
        self,
        agent_id: int,
        obs,
        action,
        reward,
        log_prob,
        state_val,
        is_terminal,
        is_interrupted,
    ):
        if agent_id not in self.buffer:
            self.buffer[agent_id] = Trajectory()

        self.buffer[agent_id].insert(obs, action, reward, log_prob, state_val)

        if is_terminal:
            self.buffer[agent_id].set_is_terminal()
        if is_interrupted:
            self.buffer[agent_id].set_is_interrupted()

        self.len += 1

    def get_flattened(self):
        obs = []
        actions = []
        rewards = []
        log_probs = []
        reward_to_gos = []

        for _, traj in self.buffer.items():
            obs.extend(traj.obs)
            actions.extend(traj.actions)
            rewards.extend(traj.rewards)
            log_probs.extend(traj.log_probs)
            reward_to_gos.extend(traj.get_reward_to_go())

        obs = torch.as_tensor(obs, dtype=torch.float32)
        actions = torch.as_tensor(actions, dtype=torch.float32)
        rewards = torch.as_tensor(rewards, dtype=torch.float32)
        log_probs = torch.as_tensor(log_probs, dtype=torch.float32)
        reward_to_gos = torch.as_tensor(reward_to_gos, dtype=torch.float32)

        return obs, actions, rewards, log_probs, reward_to_gos

    def __len__(self):
        return self.len


if __name__ == "__main__":
    trajbuff = TrajectoryBuffer()

    trajbuff.add(3, [3, 4, 5], 3, 2, 0.5, 0, 0, 3)
    trajbuff.add(3, [4, 9, 3], 2, 3, 0.4, 0, 0, 2)
    trajbuff.add(3, [9, 1, 4], 1, 5, 0.1, 0, 0, 1)
    trajbuff.add(1, [2, 1, 3], 0, 7, 0.8, 0, 0, 0)
    trajbuff.add(3, [2, 1, 3], 8, 9, 0.11, 1, 0, -1)

    print(trajbuff.get_flattened())
