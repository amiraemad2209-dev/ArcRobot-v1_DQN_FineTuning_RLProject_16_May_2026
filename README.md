## Acrobot DQN Streamlit Demo

This project demonstrates a Deep Q-Network (DQN) agent trained to solve the **Acrobot-v1** environment using Gymnasium and Stable-Baselines3.

### Features
- Load and compare **pretrained vs fine-tuned DQN models**
- Run real-time **simulation with evaluation metrics (reward + steps)**
- Visualize performance with graphs 
- Generate comparison graphs between models
- Clean UI with custom styling

### Structure
- `load_model()` → loads old/new DQN models
- `evaluate_model()` → calculates reward and steps only
- `run_simulation()` → runs episode + shows video + metrics
- Pages:
  - Demo (single model test)
  - Before vs After (comparison)
  - Graphs (performance analysis)
  - About DQN (theory)

### Key Idea
The agent learns to minimize steps (since reward = -1 per step) by swinging the pole efficiently to the target position.

### Tech Stack
Streamlit · Gymnasium · Stable-Baselines3 · DQN · Matplotlib
