from decision_steps import DecisionSteps
from terminal_steps import TerminalSteps
from trajectory_buffer import TrajectoryBuffer
from mlagents_envs.environment import UnityEnvironment, ActionTuple
from mlagents_envs.side_channel.engine_configuration_channel import (
    EngineConfigurationChannel,
)
import numpy as np


class GameInterface:
    def __init__(self, traj_buff) -> None:
        self.traj_buff = traj_buff
        self.engine_config_channel = EngineConfigurationChannel()
        self.engine_config_channel.set_configuration_parameters(time_scale=0.01)
        self.env = UnityEnvironment(
            file_name=None, seed=1, side_channels=[self.engine_config_channel]
        )
        self.env.reset()
        assert (
            len(list(self.env.behavior_specs)) == 1
        ), "All agents should belong to the same behaviour"
        self.behavior_name = list(self.env.behavior_specs)[0]

    def get_state(self):
        dec_steps, term_steps = self.env.get_steps(self.behavior_name)

        d = DecisionSteps(
            dec_steps.obs[0],
            dec_steps.reward,
            dec_steps.agent_id,
        )
        t = TerminalSteps(
            term_steps.obs[0],
            term_steps.reward,
            term_steps.agent_id,
            term_steps.interrupted,
        )
        return d, t

    def set_actions(self, actions):
        self.env.set_actions(self.behavior_name, ActionTuple(discrete=actions))

    def step(self):
        self.env.step()

    def close(self):
        self.env.close()

    def debug_random_play(self, steps=2000):
        try:
            for _ in range(steps):
                # 1. Get the current state (using your dictionary method)
                dec, term = self.get_state()

                # 2. Prepare actions for all active agents
                # We need a 2D array: (number_of_agents, number_of_branches)
                num_agents = len(dec.agent_ids)

                if num_agents > 0:
                    # one for each team
                    move_actions = np.random.randint(0, 1, size=(num_agents, 1))
                    jump_actions = np.random.randint(1, 2, size=(num_agents, 1))

                    # Combine into a single array: [[move, jump], [move, jump], ...]
                    combined_actions = np.hstack([move_actions, jump_actions])

                    # 3. Create the ActionTuple for ML-Agents
                    actions = ActionTuple()
                    actions.add_discrete(combined_actions)

                    # 4. Step the environment for each behavior
                    # Assuming all agents share the same behavior_name
                    self.env.set_actions(self.behavior_name, actions)

                # 5. Advance the simulation
                self.env.step()

                # 6. Optional: Handle terminal steps (resets happen automatically in Unity)
                if len(term.agent_ids) > 0:
                    print(
                        f"Resetting {len(term.agent_ids)} agents due to goal or timeout."
                    )
        finally:
            self.env.close()


if __name__ == "__main__":
    game = GameInterface(TrajectoryBuffer())
    game.debug_random_play()
