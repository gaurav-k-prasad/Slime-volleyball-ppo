import torch
from actor import MultiHeadActor
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.base_env import ActionTuple
import numpy as np


def evaluate(model_path, obs_dim, action_dim):
    # 1. Setup Environment (Connect to Unity Editor)
    env = UnityEnvironment()
    env.reset()
    behavior_name = list(env.behavior_specs.keys())[0]

    # 2. Load the Model
    # Note: adjust (3, 2) to match your specific action_dim
    actor = MultiHeadActor(obs_dim, action_dim[0], action_dim[1])
    checkpoint = torch.load(model_path)
    actor.load_state_dict(checkpoint["actor_state_dict"])

    # Set to evaluation mode (stops exploration)
    actor.eval()

    print("Starting Evaluation... Press Ctrl+C to stop.")

    try:
        while True:
            # Get environment state
            decision_steps, terminal_steps = env.get_steps(behavior_name)

            if len(decision_steps) > 0:
                # 1. Get observations for ALL agents (e.g., 32 agents)
                # Ensure this matches your data structure. Usually it's decision_steps.obs[0]
                obs = torch.as_tensor(decision_steps.obs[0], dtype=torch.float32)

                with torch.no_grad():
                    # actor(obs) will now return move_probs for all 32 agents
                    move_probs, jump_probs = actor(obs)

                    # Argmax will result in arrays of length 32
                    move_actions = torch.argmax(move_probs, dim=-1).cpu().numpy()
                    jump_actions = torch.argmax(jump_probs, dim=-1).cpu().numpy()

                # 2. Combine them into a (32, 2) shape
                combined_actions = np.column_stack((move_actions, jump_actions))

                # 3. Create the ActionTuple
                # Ensure it's Discrete if your Unity settings are Discrete
                action_tuple = ActionTuple(discrete=combined_actions)

                # 4. Send actions for ALL agents
                env.set_actions(behavior_name, action_tuple)

            env.step()

    except KeyboardInterrupt:
        print("\nEvaluation stopped.")
    finally:
        env.close()


if __name__ == "__main__":
    # Parameters must match your training setup
    MODEL_FILE = "final_model.pth"
    OBS_SIZE = 11
    ACT_SIZE = (3, 2)

    evaluate(MODEL_FILE, OBS_SIZE, ACT_SIZE)
