from actor import MultiHeadActor
from critic import Critic
from game_interface import GameInterface
from trajectory_buffer import TrajectoryBuffer

from torch.distributions import Categorical
import torch
import numpy as np


class PPO:
    def __init__(
        self,
        obs_dim,
        action_dim: tuple[int, int],
        steps_per_batch: int,
        n_updates_per_iterations: int,
        clip: float,
        actor_lr,
        critic_lr,
    ):
        self.traj_buff = TrajectoryBuffer()
        self.game_interface = GameInterface(self.traj_buff)
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.clip = clip
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr

        self.critic = Critic(self.obs_dim, 1)
        self.actor = MultiHeadActor(
            self.obs_dim, self.action_dim[0], self.action_dim[1]
        )
        self.actor_optim = torch.optim.Adam(
            self.actor.parameters(),
            lr=self.actor_lr,
        )
        self.critic_optim = torch.optim.Adam(
            self.critic.parameters(),
            lr=self.critic_lr,
        )
        self.steps_per_batch = steps_per_batch
        self.n_updates_per_iterations = n_updates_per_iterations

    def get_action(self, state):
        move_probs, jump_probs = self.actor(state)

        move_dist = Categorical(move_probs)
        move_action = move_dist.sample()
        move_log_prob = move_dist.log_prob(move_action)

        jump_dist = Categorical(jump_probs)
        jump_action = jump_dist.sample()
        jump_log_prob = jump_dist.log_prob(jump_action)

        # joint_prob = move_prob * jump_prob
        # log(joint_prob) = log(move_prob * jump_prob) = log(move_prob) + log(jump_prob)
        combined_log_probs = move_log_prob + jump_log_prob
        return move_action, jump_action, combined_log_probs

    def get_new_log_probs(self, batch_obs, move_batch_acts, jump_batch_acts):
        move_probs, jump_probs = self.actor(batch_obs)

        move_dist = Categorical(move_probs)
        move_log_probs = move_dist.log_prob(move_batch_acts)

        jump_dist = Categorical(jump_probs)
        jump_log_probs = jump_dist.log_prob(jump_batch_acts)

        # joint_prob = move_prob * jump_prob
        # log(joint_prob) = log(move_prob * jump_prob) = log(move_prob) + log(jump_prob)
        combined_log_probs = move_log_probs + jump_log_probs
        return combined_log_probs

    def get_momentum(
        self,
        batch_obs,
        move_batch_acts,
        jump_batch_acts,
        log_probs,
    ):
        new_log_probs = self.get_new_log_probs(
            batch_obs, move_batch_acts, jump_batch_acts
        )
        ratio = torch.exp(new_log_probs - log_probs)
        return ratio

    def get_actor_loss(
        self, batch_obs, move_batch_acts, jump_batch_acts, batch_log_probs, batch_adv
    ):
        ratio = self.get_momentum(
            batch_obs, move_batch_acts, jump_batch_acts, batch_log_probs
        )
        surr1 = ratio * batch_adv
        surr2 = torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * batch_adv

        actor_loss = (-torch.min(surr1, surr2)).mean()
        return actor_loss

    def step(self, step_no):
        decision_steps, terminal_steps = self.game_interface.get_state()

        if len(decision_steps) > 0:
            decision_obs_t = torch.as_tensor(decision_steps.obs, dtype=torch.float32)
            with torch.no_grad():
                decision_state_vals = self.critic(decision_obs_t).view(-1).cpu().numpy()
                (
                    decision_move_actions,
                    decision_jump_actions,
                    log_probs,
                ) = self.get_action(decision_obs_t)

            decision_move_actions = decision_move_actions.cpu().numpy()
            log_probs = log_probs.cpu().numpy()
            decision_jump_actions = decision_jump_actions.cpu().numpy()

            for i, agent_id in enumerate(decision_steps.agent_ids):
                self.traj_buff.add(
                    agent_id,
                    decision_steps.obs[i],
                    decision_move_actions[i],
                    decision_jump_actions[i],
                    decision_steps.rewards[i],
                    log_probs[i],
                    decision_state_vals[i],
                    False,
                    # if it's the last step in batch it's interrupted
                    step_no == self.steps_per_batch - 1,
                )
            combined_actions = np.column_stack(
                (decision_move_actions, decision_jump_actions)
            )
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
                    0,  # invalid as no need for it when terminal
                    terminal_steps.rewards[i],
                    0,  # invalid as no need for it when terminal
                    terminal_state_vals[i],
                    not terminal_steps.is_interrupted[i],
                    terminal_steps.is_interrupted[i],
                )
        self.game_interface.step()

    def play_batch(self):
        self.traj_buff.clear()
        for i in range(self.steps_per_batch):
            self.step(i)

        obs, move_actions, jump_actions, log_probs, state_vals, rtgs, advantages = (
            self.traj_buff.get_flattened()
        )

        return obs, move_actions, jump_actions, log_probs, state_vals, rtgs, advantages

    def learn(self, total_timesteps):
        t_so_far = 0
        while t_so_far < total_timesteps:
            obs, move_actions, jump_actions, log_probs, _, rtgs, advantages = (
                self.play_batch()
            )

            for _ in range(self.n_updates_per_iterations):
                state_vals = self.critic(obs).squeeze()
                actor_loss = self.get_actor_loss(
                    obs, move_actions, jump_actions, log_probs, advantages
                )
                critic_loss = torch.nn.MSELoss()(state_vals, rtgs)

                self.actor_optim.zero_grad()
                actor_loss.backward()
                self.actor_optim.step()

                self.critic_optim.zero_grad()
                critic_loss.backward()
                self.critic_optim.step()

            t_so_far += self.steps_per_batch
            print(f"timesteps: {t_so_far}")


if __name__ == "__main__":
    ppo = PPO(8, (3, 2), 1000, 6, 0.2, 1e-3, 1e-3)
    ppo.learn(10000)
