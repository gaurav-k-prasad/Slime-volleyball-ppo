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
        entropy_coeff=0.01,
        move_entropy_bonus=1,
        jump_entropy_bonus=1.5,
        model_path=None,
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.traj_buff = TrajectoryBuffer()
        self.game_interface = GameInterface(self.traj_buff)
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.clip = clip
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.entropy_coeff = entropy_coeff
        self.m_entropy_bonus = move_entropy_bonus
        self.j_entropy_bonus = jump_entropy_bonus

        self.critic = Critic(self.obs_dim, 1).to(self.device)
        self.actor = MultiHeadActor(
            self.obs_dim, self.action_dim[0], self.action_dim[1]
        ).to(self.device)
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

        if model_path:
            self.load(model_path)

    def save(self, path="ppo_model.pth"):
        torch.save(
            {
                "actor_state_dict": self.actor.state_dict(),
                "critic_state_dict": self.critic.state_dict(),
                "actor_optimizer_state_dict": self.actor_optim.state_dict(),
                "critic_optimizer_state_dict": self.critic_optim.state_dict(),
            },
            path,
        )
        print(f"Model saved to {path}")

    def load(self, path):
        checkpoint = torch.load(path)
        self.actor.load_state_dict(checkpoint["actor_state_dict"])
        self.critic.load_state_dict(checkpoint["critic_state_dict"])
        self.actor_optim.load_state_dict(checkpoint["actor_optimizer_state_dict"])
        self.critic_optim.load_state_dict(checkpoint["critic_optimizer_state_dict"])
        print(f"Model loaded from {path}")

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
        m_entropy = move_dist.entropy().mean()
        j_entropy = jump_dist.entropy().mean()
        total_entropy = (self.m_entropy_bonus * m_entropy) + (
            self.j_entropy_bonus * j_entropy
        )

        return combined_log_probs, total_entropy

    def get_momentum(
        self,
        batch_obs,
        move_batch_acts,
        jump_batch_acts,
        log_probs,
    ):
        new_log_probs, entropy = self.get_new_log_probs(
            batch_obs, move_batch_acts, jump_batch_acts
        )
        ratio = torch.exp(new_log_probs - log_probs)
        return ratio, entropy

    def get_actor_loss(
        self, batch_obs, move_batch_acts, jump_batch_acts, batch_log_probs, batch_adv
    ):
        ratio, entropy = self.get_momentum(
            batch_obs, move_batch_acts, jump_batch_acts, batch_log_probs
        )
        surr1 = ratio * batch_adv
        surr2 = torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * batch_adv

        actor_loss = (-torch.min(surr1, surr2)).mean()
        return actor_loss - (entropy * self.entropy_coeff)

    def step(self, step_no):
        decision_steps, terminal_steps = self.game_interface.get_state()

        if len(decision_steps) > 0:
            decision_obs_t = torch.as_tensor(
                decision_steps.obs, dtype=torch.float32
            ).to(self.device)
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
            terminal_obs_t = torch.as_tensor(
                terminal_steps.obs, dtype=torch.float32
            ).to(self.device)

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
            self.traj_buff.get_flattened(self.device)
        )

        return obs, move_actions, jump_actions, log_probs, state_vals, rtgs, advantages

    def learn(self, start, more_timesteps):
        t_so_far = start
        total_timesteps = start + more_timesteps
        i = 1
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

            print(f"timesteps = {t_so_far}, {i = }")
            if i % 10 == 0 and i:
                self.save(f"models/model_at_{t_so_far}.pth")

            i += 1


if __name__ == "__main__":
    ppo = PPO(
        11,
        (3, 2),
        2000,
        6,
        0.2,
        1e-3,
        1e-3,
        entropy_coeff=0.05,
    )
    try:
        ppo.learn(660_000, 100_000)
    finally:
        ppo.game_interface.close()
