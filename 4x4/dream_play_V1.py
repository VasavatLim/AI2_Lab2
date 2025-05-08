import torch
import gymnasium as gym
import numpy as np
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor
import matplotlib.pyplot as plt
from collections import defaultdict

from game import register_custom_frozen_lake, generate_frozen_lake_env, CUSTOM_FROZEN_LAKE_ID
from model.jepa import CNNEncoder, Predictor, CNNDecoder, JEPA

# === Register and create custom FrozenLake environment ===
register_custom_frozen_lake()
env = generate_frozen_lake_env(CUSTOM_FROZEN_LAKE_ID, map_size=4)
obs, _ = env.reset()

# === Image transformation pipeline ===
transform = Compose([Resize(64), ToTensor()])

# === Load JEPA model ===
encoder = CNNEncoder([3, 4, 8, 16])
predictor = Predictor(encoder_dim=1024)
decoder = CNNDecoder([16, 8, 4, 3], embedding_img_size=(8, 8))
model = JEPA(encoder=encoder, predictor=predictor, debug_decoder=decoder)

# Load checkpoint
ckpt = torch.load("checkpoints/jepa_with_vicreg.ckpt", map_location="cpu")
state_dict = {k.replace("model.", ""): v for k, v in ckpt["state_dict"].items()}
model.load_state_dict(state_dict, strict=False)
model.eval()

# === Fix the goal color from the bottom-right tile ===
raw_img = env.render()
goal_tile_rgb = raw_img[-1, -1]  # Sample bottom-right pixel
goal_color = goal_tile_rgb / 255.0  # Normalize to [0, 1]
print(f"[INFO] Fixed goal color from bottom-right: {goal_color}")

# === Scoring function for predicted images ===
def score_image(img_tensor):
    img_np = img_tensor.squeeze().detach().numpy()  # [3, H, W]
    img_np = np.transpose(img_np, (1, 2, 0))  # [H, W, C]
    diff = np.abs(img_np - goal_color)
    distance = diff.sum(axis=2)
    return -np.mean(distance)  # Lower distance = closer to goal

# === World model control loop ===
action_space = [0, 1, 2, 3]  # Left, Down, Right, Up
action_names = ["Left", "Down", "Right", "Up"]
max_steps = 60
done = False
state_visit_count = defaultdict(int)
last_obs = None
stuck_counter = 0
max_stuck = 20
step = 0
recent_actions = []
max_recent = 2

while not done and step < max_steps:
    raw_img = env.render()
    x = transform(Image.fromarray(raw_img))
    s_x = model.encoder(x.unsqueeze(0))

    best_score = -np.inf
    best_action = 0

    print(f"Step {step} dreamed scores:")
    for a in action_space:
        a_tensor = torch.tensor([a], dtype=torch.long)
        s_y_pred = model.predictor(s_x, a_tensor)
        recon = model.debug_decoder(s_y_pred)
        score = score_image(recon)

        penalty = 0.1 if recent_actions.count(a) > 0 else 0.0
        novelty_bonus = 0.05 if state_visit_count[obs] == 0 else -0.05
        state_penalty = 0.05 * state_visit_count[obs]

        adjusted_score = score - penalty - state_penalty + novelty_bonus
        print(f"  {a} ({action_names[a]}): {adjusted_score:.4f} (penalty: {penalty}, novelty: {novelty_bonus}, state_penalty: {state_penalty:.2f})")

        if adjusted_score > best_score:
            best_score = adjusted_score
            best_action = a

    obs, reward, done, truncated, info = env.step(best_action)

    recent_actions.append(best_action)
    if len(recent_actions) > max_recent:
        recent_actions.pop(0)

    state_visit_count[obs] += 1
    stuck_counter = stuck_counter + 1 if state_visit_count[obs] > 2 else 0

    print(f"\nStep {step}: action={best_action} ({action_names[best_action]}), reward={reward}, done={done}")

    plt.imshow(raw_img)
    plt.title(f"Action: {action_names[best_action]}")
    plt.axis('off')
    plt.pause(0.5)

    if stuck_counter >= max_stuck:
        print("Agent is stuck revisiting states too often — breaking loop.")
        break

    if done and reward == 0:
        print("Agent fell into a hole ❄️ — game over.")
        break

    if reward == 1:
        print("🌟 Goal reached!")
        break

    step += 1

env.close()
plt.close()
