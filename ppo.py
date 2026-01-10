from actor import Actor
from critic import Critic
from game_interface import GameInterface
from trajectory_buffer import TrajectoryBuffer
from torch.distributions import Categorical
import torch
import numpy as np


class PPO:
    def __init__(self, obs_dim, action_dim, steps_per_batch: int):
        self.traj_buff = TrajectoryBuffer()
        self.game_interface = GameInterface(self.traj_buff)
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        self.critic = Critic(self.obs_dim, 1)
        self.actor = Actor(self.obs_dim, self.action_dim)

        self.steps_per_batch = steps_per_batch

    def get_action(self, state):
        probs = self.actor(state)
        dist = Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob

    def step(self, step_no):
        decision_steps, terminal_steps = self.game_interface.get_state()

        if len(decision_steps) > 0:
            decision_obs_t = torch.as_tensor(decision_steps.obs, dtype=torch.float32)
            with torch.no_grad():
                decision_state_vals = self.critic(decision_obs_t).view(-1).cpu().numpy()
                decision_actions, decision_log_probs = self.get_action(decision_obs_t)

            decision_actions = decision_actions.cpu().numpy()
            decision_log_probs = decision_log_probs.cpu().numpy()

            for i, agent_id in enumerate(decision_steps.agent_ids):
                self.traj_buff.add(
                    agent_id,
                    decision_steps.obs[i],
                    decision_actions[i],
                    decision_steps.rewards[i],
                    decision_log_probs[i],
                    decision_state_vals[i],
                    False,
                    # if it's the last step in batch it's interrupted
                    step_no == self.steps_per_batch - 1,
                )
            jump_dummy = np.zeros_like(decision_actions)
            combined_actions = np.column_stack((decision_actions, jump_dummy))
            self.game_interface.set_actions(combined_actions)

        if len(terminal_steps) > 0:
            terminal_obs_t = torch.as_tensor(terminal_steps.obs, dtype=torch.float32)

            with torch.no_grad():
                terminal_state_vals = self.critic(terminal_obs_t).view(-1).cpu().numpy()

            for i, agent_id in enumerate(terminal_steps.agent_ids):
                self.traj_buff.add(
                    agent_id,
                    terminal_steps.obs[i],
                    0,  # invalid as no need for it when terminal
                    terminal_steps.rewards[i],
                    0,  # invalid as no need for it when terminal
                    terminal_state_vals[i],
                    not terminal_steps.is_interrupted[i],
                    terminal_steps.is_interrupted[i],
                )
        self.game_interface.step()

    def play_batch(self):
        for i in range(self.steps_per_batch):
            self.step(i)

        obs, actions, rewards, log_probs, reward_to_gos, state_vals = (
            self.traj_buff.get_flattened(self.critic)
        )

        print("Batch completed")


if __name__ == "__main__":
    ppo = PPO(8, 3, 300)
    ppo.play_batch()
    ppo.game_interface.close()
