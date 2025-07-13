import warnings
warnings.filterwarnings("ignore")

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("Agg")

from loguru import logger
import os

import gymnasium as gym
from gymnasium.utils.save_video import save_video

from agent import Agent

os.makedirs("videos", exist_ok=True)

env = gym.make("CartPole-v1", render_mode="rgb_array_list")

agent = Agent(input_dims=env.observation_space.shape, n_actions=env.action_space.n)

n_episodes = 1000
scores = []

VIDEO_RECORD_INTERVAL = 100
RECORD_LAST_N_EPISODES = 5
RECORD_BEST_EPISODES = True
best_score = -float('inf')

def should_record_video(episode_idx, score, best_score):
    if episode_idx % VIDEO_RECORD_INTERVAL == 0:
        return True, "interval"
    
    if episode_idx >= n_episodes - RECORD_LAST_N_EPISODES:
        return True, "final"
    
    if RECORD_BEST_EPISODES and episode_idx > 50 and score > best_score:
        return True, "best"
    
    return False, None

for idx in range(n_episodes):
    
    done = False
    state, _ = env.reset()
    total_reward = 0
    
    record_this_episode, record_reason = should_record_video(idx, best_score, best_score)

    while not done:
        action = agent.choose_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        agent.store_transition(state, action, reward, next_state, done)
        agent.learn()

        state = next_state
        total_reward += reward

    if total_reward > best_score:
        best_score = total_reward
        if not record_this_episode:
            record_this_episode, record_reason = should_record_video(idx, total_reward, best_score - 1)

    if record_this_episode:
        try:
            video_name = f"episode_{idx:04d}_{record_reason}_reward_{total_reward:.0f}"
            save_video(
                frames=env.render(),
                video_folder="videos",
                name_prefix=video_name,
                fps=env.metadata["render_fps"]
            )
            logger.info(f"Video saved: {video_name} (reason: {record_reason})")
        except Exception as e:
            logger.error(f"Failed to save video for episode {idx}: {e}")

    scores.append(total_reward)
    
    if idx % 10 == 0 or record_this_episode:
        logger.info(f"EPISODE {idx}, Reward: {total_reward}, Best: {best_score}, Epsilon: {agent.epsilon:.3f}")

plt.figure(figsize=(12, 8))

plt.subplot(2, 1, 1)
plt.plot(scores, alpha=0.7, label='Episode Reward')
plt.axhline(y=best_score, color='r', linestyle='--', label=f'Best Score: {best_score:.1f}')

window_size = 50
if len(scores) >= window_size:
    moving_avg = []
    for i in range(len(scores)):
        if i >= window_size - 1:
            moving_avg.append(sum(scores[i-window_size+1:i+1]) / window_size)
        else:
            moving_avg.append(sum(scores[:i+1]) / (i+1))
    plt.plot(moving_avg, color='red', linewidth=2, label=f'Moving Average ({window_size} episodes)')

plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("DQN CartPole Performance")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('training_results.png', dpi=300, bbox_inches='tight')
plt.close()

logger.success("Training completed!")
logger.info(f"Final performance: Best reward = {best_score}")
logger.info(f"Graph saved as 'training_results.png'")
logger.info(f"Videos saved in 'videos/' directory")

video_files = [f for f in os.listdir("videos") if f.endswith('.mp4')]
if video_files:
    logger.info(f"Recorded {len(video_files)} videos:")
    for video in sorted(video_files):
        logger.info(f"  - {video}")
else:
    logger.warning("No videos were recorded!")