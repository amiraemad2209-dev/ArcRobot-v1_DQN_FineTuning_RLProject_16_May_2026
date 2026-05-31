import streamlit as st
import gymnasium as gym
import imageio.v2 as imageio
from stable_baselines3 import DQN
import sys
from huggingface_sb3 import load_from_hub
import numpy as np 
from numpy import random
import matplotlib.pyplot as plt

np.random.seed(42)
random.seed(42)


# ─────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Acrobot AI Demo",
    layout="centered"
)

# ─────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>

/* ===== GLOBAL BACKGROUND ===== */
.stApp {
    background: radial-gradient(circle at top, #0f172a, #020617);
    color: #e2e8f0;
    font-family: 'Segoe UI', sans-serif;
}

/* ===== TITLES ===== */
h1 {
    color: #38bdf8;
    text-align: center;
    text-shadow: 0 0 12px rgba(56, 189, 248, 0.6);
    font-weight: 700;
}

h3 {
    color: #94a3b8;
    text-align: center;
    font-weight: 400;
}

/* ===== METRIC CARDS ===== */
div[data-testid="stMetric"] {
    background: rgba(255, 255, 255, 0.06);
    border: 1px solid rgba(255, 255, 255, 0.12);
    padding: 16px;
    border-radius: 18px;
    backdrop-filter: blur(12px);
    box-shadow: 0 8px 20px rgba(0, 0, 0, 0.3);
    transition: all 0.3s ease;
}

div[data-testid="stMetric"]:hover {
    transform: translateY(-3px);
    box-shadow: 0 12px 25px rgba(56, 189, 248, 0.25);
}

/* ===== METRIC VALUES ===== */
[data-testid="stMetricValue"] {
    color: #22c55e;
    font-size: 30px;
    font-weight: bold;
}

/* ===== BUTTONS ===== */
.stButton button {
    background: linear-gradient(90deg, #38bdf8, #6366f1);
    color: white;
    font-weight: 600;
    border-radius: 14px;
    width: 100%;
    padding: 12px;
    border: none;
    transition: all 0.3s ease;
    box-shadow: 0 4px 12px rgba(56, 189, 248, 0.2);
}

.stButton button:hover {
    transform: scale(1.05);
    box-shadow: 0 0 20px rgba(56, 189, 248, 0.5);
}

/* ===== SIDEBAR ===== */
section[data-testid="stSidebar"] {
    background: #0b1220;
}

/* ===== VIDEO ===== */
video {
    border-radius: 15px;
    box-shadow: 0 8px 20px rgba(0,0,0,0.4);
}

</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Titles
# ─────────────────────────────────────────────
page = st.sidebar.radio(
    "Navigation",
    [
        "Demo",
        "Before vs After",
        "Graphs",
        "About DQN"
    ]
)

# =====================================================
# FUNCTION OF MODEL LOADER 
# =====================================================

@st.cache_resource
def load_model(model_type="new"):

    if model_type == "old":

        sys.modules["gym"] = gym

        ckpt = load_from_hub(
            repo_id="sb3/dqn-Acrobot-v1",
            filename="dqn-Acrobot-v1.zip",
        )

        env = gym.make("Acrobot-v1")

        custom_objects = {
            "observation_space": env.observation_space,
            "action_space": env.action_space
        }

        return DQN.load(
            ckpt,
            env=env,
            custom_objects=custom_objects
        )

    elif model_type == "new":

        return DQN.load(
            "acrobot_final_fine_tuned.zip"
        )

# =====================================================
# FUNCTION OF EVALUATION 
# =====================================================
def evaluate_model(model):

    env = gym.make("Acrobot-v1")
    obs, _ = env.reset(seed=42)
    env.action_space.seed(42)

    done = False
    total_reward = 0
    steps = 0

    while not done:

        action, _ = model.predict(obs, deterministic=True)

        obs, reward, terminated, truncated, _ = env.step(action)

        done = terminated or truncated

        total_reward += reward
        steps += 1

    env.close()

    return total_reward, steps


# =====================================================
# FUNCTION OF  SIMULATION , SHOW VIDEO & Metrics
# =====================================================

def run_simulation(model, video_path, title):

    st.subheader(title)

    # ==============================
    # 1) CALL EVALUATION HERE
    # ==============================
    total_reward, steps = evaluate_model(model)

    # ==============================
    # 2) SHOW VIDEO
    # ==============================
    col1, col2, col3 = st.columns([1, 3, 1])

    with col2:
        st.video(video_path)

    # ==============================
    # 3) METRICS
    # ==============================
    col1, col2 = st.columns(2)

    col1.metric("Total Reward", f"{total_reward:.2f}")
    col2.metric("Steps", steps)

    # ==============================
    # 4) ANALYSIS
    # ==============================
    if total_reward >= -100:
        st.success("Excellent Performance ")

    elif total_reward >= -250:
        st.warning("Moderate Performance ")

    else:
        st.error("Weak Performance ")

    return total_reward, steps



# =====================================================
# 1- DEMO PAGE
# =====================================================

if page == "Demo":

    st.title("Acrobot AI Agent")
    st.subheader("Deep Q-Network Swing-Up Controller")

    st.markdown("""
    This application demonstrates a **Deep Q-Network (DQN)** agent trained to solve the **Acrobot-v1** reinforcement learning environment.

    The agent learns how to generate momentum and swing the robotic arm upward efficiently using intelligent torque control.
    """)

    # -------------------------------------------------
    # Load Model
    # -------------------------------------------------
    
    try:
        model = load_model("new")
        st.sidebar.success("Fine-Tuned Acrobot Model Loaded")
    except Exception:
        st.sidebar.error("Model file not found.")
        st.stop()

    # -------------------------------------------------
    # Run Simulation
    # -------------------------------------------------
    if st.button("▶ Start Acrobot Simulation"):

        with st.spinner("AI agent is balancing momentum..."):

            run_simulation(
            model=model,
            video_path="acrobot_expert_performance.mp4",
            title="Fine-Tuned DQN Performance"
        )

# =========================================
# 2- BEFORE VS AFTER PAGE
# =========================================

elif page == "Before vs After":

    st.title("Before vs After Training")

    col1, col2 = st.columns(2)

    # =========================================
    # ORIGINAL MODEL
    # =========================================
    with col1:

        old_model = load_model("old")

        st.subheader("Original DQN")

        # Video only
        st.video("acrobot_pretrained.mp4")

        # Evaluation (numbers only)
        old_reward, old_steps = evaluate_model(old_model)

        st.metric("Total Reward", f"{old_reward:.2f}")
        st.metric("Steps", old_steps)


    # =========================================
    # FINE-TUNED MODEL
    # =========================================
    with col2:

        new_model = load_model("new")

        st.subheader("Fine-Tuned DQN")

        # Video only
        st.video("acrobot_expert_performance.mp4")

        # Evaluation (numbers only)
        new_reward, new_steps = evaluate_model(new_model)

        st.metric("Total Reward", f"{new_reward:.2f}")
        st.metric("Steps", new_steps)


# =====================================================
# 4- GRAPHS
# =====================================================
elif page == "Graphs":

    st.title("Performance Analysis Dashboard")

    with st.spinner("Loading models..."):

        old_model = load_model("old")
        new_model = load_model("new")

        old_reward, old_steps = evaluate_model(old_model)
        new_reward, new_steps = evaluate_model(new_model)

    import matplotlib.pyplot as plt

    labels = ["Old DQN", "Fine-Tuned DQN"]

    steps = [old_steps, new_steps]
    rewards = [old_reward, new_reward]

    fig, ax = plt.subplots(2, 2, figsize=(12, 8))

    # =================================================
    # 1. LINE - Steps Trend
    # =================================================
    ax[0, 0].plot(labels, steps, marker='o', linewidth=3, color='#3b82f6')
    ax[0, 0].set_title("Steps Trend")
    ax[0, 0].set_ylabel("Steps")
    ax[0, 0].grid(True, alpha=0.3)

    for i, v in enumerate(steps):
        ax[0, 0].text(i, v, str(v), ha='center', fontsize=10)

    # =================================================
    # 2. LINE - Reward Trend
    # =================================================
    ax[0, 1].plot(labels, rewards, marker='o', linewidth=3, color='#22c55e')
    ax[0, 1].set_title("Reward Trend")
    ax[0, 1].set_ylabel("Reward")
    ax[0, 1].axhline(0, color="gray", linestyle="--", linewidth=1)
    ax[0, 1].grid(True, alpha=0.3)

    for i, v in enumerate(rewards):
        ax[0, 1].text(i, v, str(v), ha='center', fontsize=10)

    # =================================================
    # 3. BAR CHART - Steps Comparison (NEW)
    # =================================================
    ax[1, 0].bar(labels, steps, color=['#f59e0b', '#3b82f6'])
    ax[1, 0].set_title("Steps Comparison (Bar Chart)")
    ax[1, 0].set_ylabel("Steps")
    ax[1, 0].grid(axis='y', alpha=0.3)

    for i, v in enumerate(steps):
        ax[1, 0].text(i, v, str(v), ha='center', va='bottom')

    # =================================================
    # 4. STEP EFFICIENCY VISUALIZATION (NEW STYLE)
    # =================================================

    efficiency = [
    old_steps,
    new_steps
     ]

    ax[1, 1].plot(labels, efficiency, marker='o', linewidth=4, color='#ef4444')
    ax[1, 1].fill_between(labels, efficiency, color='#ef4444', alpha=0.2)

    ax[1, 1].set_title("Step Efficiency Drop (Lower is Better)")
    ax[1, 1].set_ylabel("Steps")
    ax[1, 1].grid(True, alpha=0.3)

    for i, v in enumerate(efficiency):  
        ax[1, 1].text(i, v, str(v), ha='center', fontsize=10)

    plt.tight_layout()
    st.pyplot(fig)

    # =================================================
    # IMPROVEMENT METRICS
    # =================================================
    improvement_steps = ((old_steps - new_steps) / old_steps) * 100
    improvement_reward = ((new_reward - old_reward) / abs(old_reward)) * 100

    st.markdown("---")

    st.subheader("Performance Summary")

    st.metric("Old Steps", old_steps)
    st.metric("New Steps", new_steps)
    st.metric("Old Reward", old_reward)
    st.metric("New Reward", new_reward)

    st.metric("Improvement in Steps (%)", f"{improvement_steps:.1f}%")
    st.metric("Improvement in Reward (%)", f"{improvement_reward:.1f}%")

    # =================================================
    # SUMMARY BOX
    # =================================================
    st.info(f"""
    📊 Summary:

    ✔ The Fine-Tuned model uses fewer steps.
    ✔ The reward improved compared to the original DQN.
    ✔ Overall performance shows better efficiency and control.
    """)


# =====================================================
# 4- ABOUT DQN PAGE
# =====================================================
elif page == "About DQN":

    st.title("About DQN")

    st.markdown("""
    ## What is Reinforcement Learning?

    Reinforcement Learning (RL) is a machine learning approach where an agent learns by interacting with an environment.

    ## What is DQN?

    Deep Q-Network (DQN) combines:

    - Q-Learning
    - Deep Neural Networks

    to estimate the best action for each state.

    ## Acrobot Goal

    The goal is to swing the robot arm upward until it reaches the target height.

    ## Reward Function

    The environment gives a reward of -1 at every step.

    Therefore, the agent learns to solve the task in as few steps as possible.
    """)


# =====================================================
# FOOTER
# =====================================================
st.markdown("---")

st.caption(
    "Powered by Stable-Baselines3 & Gymnasium | Reinforcement Learning Project 2026"
)