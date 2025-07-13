# DQN CartPole - Reinforcement Learning

A Deep Q-Network (DQN) implementation for solving the CartPole-v1 environment using PyTorch and Gymnasium.

## 🎯 Overview

This project implements a Deep Q-Network agent to learn the CartPole balancing task. The agent uses experience replay and a target network to stabilize training, following the DQN algorithm introduced by DeepMind.

## 🚀 Features

- **Deep Q-Network (DQN)** implementation with PyTorch
- **Experience Replay** buffer for stable learning
- **Target Network** for reduced correlation in Q-value updates
- **Epsilon-greedy exploration** with decay
- **Video Recording** of training episodes (best episodes, intervals, and final episodes)
- **Training Visualization** with performance plots
- **Logging** with detailed training information

## 📋 Requirements

- [uv](https://docs.astral.sh/uv/) - Python package and project manager

## 🛠️ Installation

1. Clone the repository:

```bash
git clone https://github.com/vnniciusg/rl-cart-pole.git
cd rl-cart-pole
```

2. Install dependencies using uv (recommended):

```bash
uv sync
```

Or using pip:

```bash
pip install gymnasium[classic-control,other] torch matplotlib loguru numpy
```

## 🎮 Usage

Run the training script:

```bash
uv run main.py
```

The script will:

- Train a DQN agent for 1000 episodes
- Record videos of notable episodes (best performance, intervals, final episodes)
- Save training results as a plot (`training_results.png`)
- Log training progress and statistics

## 📁 Project Structure

```
rl-cart-pole/
├── main.py          # Main training script
├── agent.py         # DQN Agent implementation
├── dqn.py          # Deep Q-Network architecture
├── pyproject.toml   # Project dependencies
├── README.md        # This file
├── videos/          # Recorded training videos
```

## 🧠 Algorithm Details

### Deep Q-Network (DQN)

- **Architecture**: 3-layer fully connected network (128-128 hidden units)
- **Input**: 4-dimensional state space (cart position, cart velocity, pole angle, pole angular velocity)
- **Output**: 2 Q-values (left and right actions)
- **Activation**: ReLU for hidden layers

### Training Parameters

- **Episodes**: 1000
- **Learning Rate**: 1e-3
- **Gamma (Discount Factor)**: 0.99
- **Epsilon Decay**: 0.995 (starts at 1.0, minimum 0.01)
- **Memory Size**: 1000 transitions
- **Batch Size**: 64
- **Target Network Update**: Every 1000 steps

### Video Recording

- **Interval Videos**: Every 100 episodes
- **Best Episodes**: When agent achieves new best score (after episode 50)
- **Final Episodes**: Last 5 episodes of training

## 📊 Results

The agent typically learns to balance the pole for the maximum 500 steps within 200-300 episodes. Training results include:

- Episode rewards over time
- Moving average performance
- Epsilon decay visualization
- Best episode recordings

## 🎥 Video Output

Training videos are automatically saved in the `videos/` directory with descriptive names:

- `episode_XXXX_interval_reward_YYY.mp4` - Interval recordings
- `episode_XXXX_best_reward_YYY.mp4` - New best performance
- `episode_XXXX_final_reward_YYY.mp4` - Final episodes

## 📈 Monitoring

The script provides detailed logging including:

- Episode rewards and best scores
- Current epsilon value
- Video recording notifications
- Training completion summary

## 🔧 Customization

You can modify training parameters in the respective files:

- **Agent parameters**: `agent.py` (learning rate, epsilon decay, memory size)
- **Network architecture**: `dqn.py` (hidden layer sizes, activation functions)
- **Training settings**: `main.py` (number of episodes, video recording frequency)

## 📚 References

- [PyTorch DQN Tutorial](https://docs.pytorch.org/tutorials/intermediate/reinforcement_q_learning.html)
- [Gymnasium CartPole Environment](https://gymnasium.farama.org/environments/classic_control/cart_pole/)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
