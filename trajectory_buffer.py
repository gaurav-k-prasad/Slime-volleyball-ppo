import torch
from trajectory import Trajectory
import numpy as np


class TrajectoryBuffer:
    def __init__(self):
        self.buffer: dict[int, list[Trajectory]] = {}
        self.len = 0

    def add(
        self,
        agent_id: int,
        obs,
        move_action,
        jump_action,
        reward,
        log_prob,
        state_val,
        is_terminal,
        is_interrupted,
    ):
        if agent_id not in self.buffer:
            self.buffer[agent_id] = []

        last_traj = self.buffer[agent_id][-1] if self.buffer[agent_id] else None

        if last_traj is None or last_traj.is_interrupted or last_traj.is_terminal:
            last_traj = Trajectory()
            self.buffer[agent_id].append(last_traj)

        last_traj.insert(
            obs,
            move_action,
            jump_action,
            reward,
            log_prob,
            state_val,
        )

        if is_terminal:
            last_traj.set_is_terminal()
        if is_interrupted:
            last_traj.set_is_interrupted()

        self.len += 1

    def get_flattened(self):
        obs = []
        move_actions = []
        jump_actions = []
        log_probs = []
        state_vals = []
        advantages = []
        rtgs = []

        for trajs in self.buffer.values():
            for traj in trajs:
                rtg = traj.get_reward_to_go()

                obs.extend(traj.obs)
                move_actions.extend(traj.move_actions)
                jump_actions.extend(traj.jump_actions)
                log_probs.extend(traj.log_probs)
                rtgs.extend(rtg)
                advantages.extend(traj.get_advantages(rtg))
                state_vals.extend(traj.state_vals)

        obs = torch.as_tensor(np.array(obs), dtype=torch.float32)
        move_actions = torch.as_tensor(move_actions, dtype=torch.float32)
        jump_actions = torch.as_tensor(jump_actions, dtype=torch.float32)
        log_probs = torch.as_tensor(log_probs, dtype=torch.float32)
        state_vals = torch.as_tensor(state_vals, dtype=torch.float32)
        rtgs = torch.as_tensor(rtgs, dtype=torch.float32)
        advantages = torch.as_tensor(advantages, dtype=torch.float32)

        return (
            obs,
            move_actions,
            jump_actions,
            log_probs,
            state_vals,
            rtgs,
            advantages,
        )

    def clear(self):
        self.buffer.clear()

    def __len__(self):
        return self.len


if __name__ == "__main__":
    trajbuff = TrajectoryBuffer()

    # trajbuff.add(3, [3, 4, 5], 3, 2, 0.5, 0, 0, 0)
    # trajbuff.add(3, [4, 9, 3], 2, 3, 0.4, 0, 0, 0)
    # trajbuff.add(3, [9, 1, 4], 1, 5, 0.1, 0, 0, 0)
    # trajbuff.add(1, [2, 1, 3], 0, 7, 0.8, 0, 0, 0)
    # trajbuff.add(3, [2, 1, 3], 8, 9, 0.11, 1, 0, 0)
