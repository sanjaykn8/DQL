import os
import torch
import random
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from stable_baselines3.common.atari_wrappers import MaxAndSkipEnv
from stable_baselines3.common.buffers import ReplayBuffer
import ale_py
import gymnasium as gym
from gymnasium import spaces
import pickle

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DQN(nn.Module):
    def __init__(self, nb_actions):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1), nn.ReLU(),
            nn.Flatten(), nn.Linear(3136, 512), nn.ReLU(),
            nn.Linear(512, nb_actions),
        )

    def forward(self, x):
        return self.network(x / 255.0)


def load_model_state_dict(path: str, nb_actions: int, device: torch.device):
    """
    Try to load a state_dict. If the file is a full-model object, extract its state_dict,
    save it as state_dict for future runs, and return a model loaded with that state_dict.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    # Try to load the file
    try:
        ckpt = torch.load(path, map_location=device)
    except Exception as e:
        # If torch 2.6+ weights_only problem arises, try a fallback that allows unpickling the full model.
        # This is a trusted-file fallback. It will extract state_dict and save it.
        try:
            torch.serialization.add_safe_globals([DQN])
            ckpt = torch.load(path, map_location=device, weights_only=False)
        except Exception as e2:
            raise RuntimeError(f"Unable to load checkpoint: {e}\nFallback load failed: {e2}")

    # If ckpt looks like a state_dict
    if isinstance(ckpt, dict) and any(isinstance(v, torch.Tensor) for v in ckpt.values()):
        state = ckpt
        model = DQN(nb_actions).to(device)
        model.load_state_dict(state)
        return model

    # If ckpt is a dict wrapper with 'state_dict' key
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = ckpt["state_dict"]
        model = DQN(nb_actions).to(device)
        model.load_state_dict(state)
        return model

    # If ckpt is a full nn.Module instance
    if isinstance(ckpt, nn.Module):
        model = ckpt.to(device)
        # save its state_dict for future easy loading
        state_path = path + ".state_dict.pth"
        try:
            torch.save(model.state_dict(), state_path)
        except Exception:
            pass
        return model

    # Last attempt: try to unpickle full object and handle it
    try:
        torch.serialization.add_safe_globals([DQN])
        obj = torch.load(path, map_location=device, weights_only=False)
        if isinstance(obj, nn.Module):
            model = obj.to(device)
            try:
                torch.save(model.state_dict(), path + ".state_dict.pth")
            except Exception:
                pass
            return model
    except Exception:
        pass

    raise RuntimeError("Checkpoint format not recognised or unsupported.")

def preprocess_obs(obs):
    arr = np.asarray(obs)            # handle LazyFrames / arrays
    # remove singleton dims (common when keep_dim=True)
    while arr.ndim > 4:
        arr = np.squeeze(arr, axis=0) if arr.shape[0] == 1 else np.squeeze(arr, axis=-1)
    if arr.ndim == 4 and arr.shape[-1] == 1:
        arr = np.squeeze(arr, axis=-1)
    # now arr should be 3D: either (4,84,84) or (84,84,4)
    if arr.ndim != 3:
        raise RuntimeError(f"Unexpected obs shape after squeeze: {arr.shape}")
    if arr.shape[0] == 4:
        ch_first = arr
    elif arr.shape[2] == 4:
        ch_first = arr.transpose(2, 0, 1)
    else:
        # fallback: attempt to move last axis to channel axis
        ch_first = np.moveaxis(arr, -1, 0)
    tensor = torch.from_numpy(ch_first).unsqueeze(0).to(DEVICE).float()  # [1,4,84,84]
    return tensor

def evaluation(reward_value, episodes=10, render_mode="rgb_array"):
    # Create gym env and wrappers
    env_display = gym.make("ALE/Breakout-v5", render_mode=render_mode)
    env_display = gym.wrappers.RecordVideo(
        env_display,
        episode_trigger=lambda num: True,  # record every episode
        video_folder="saved-video-folder",
        name_prefix="struggle_video-",
    )
    env_display = gym.wrappers.RecordEpisodeStatistics(env_display)
    env_display = gym.wrappers.ResizeObservation(env_display, (84, 84))
    env_display = gym.wrappers.GrayscaleObservation(env_display, keep_dim=True)
    env_display = gym.wrappers.FrameStackObservation(env_display, 4)
    env_display = MaxAndSkipEnv(env_display, skip=4)

    # Load model (state_dict or full model fallback)
    best_model_path = f"./best_models/best_model_{reward_value}"
    model = load_model_state_dict(best_model_path, env_display.action_space.n, DEVICE)
    model.eval()

    try:
        for ep in range(episodes):
            obs, info = env_display.reset()
            terminated = False
            truncated = False
            done_flag = False
            total_rewards = 0
            steps = 0

            while not (terminated or truncated):
                # prepare tensor: expect shape [4,84,84] -> convert to [1,4,84,84]
                obs_tensor = preprocess_obs(obs)
                with torch.no_grad():
                    q_values = model(obs_tensor)
                    action = int(q_values.argmax(dim=1).item())

                # force FIRE at episode start to launch ball if needed
                if steps == 0:
                    action = 1

                obs, reward, terminated, truncated, info = env_display.step(action)
                total_rewards += float(reward)
                steps += 1

                if reward != 0:
                    print(f"Episode {ep} - Steps: {steps}, Reward: {reward}")

            print(f"Episode {ep} finished. Total rewards: {total_rewards}")

    finally:
        # ensure video flush
        env_display.close()
        print("Evaluation completed. Videos in saved-video-folder/ (if any).")


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    # Register ALE and quick smoke
    gym.register_envs(ale_py)

    # Evaluation: set to the numeric suffix of your saved best model file.
    # Example: if file is "./best_models/best_model_67.0" set best_rewards = 67.0
    best_rewards = 76.0
    evaluation(best_rewards, episodes=10)
