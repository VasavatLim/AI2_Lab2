from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms import Compose, Resize, ToTensor
from PIL import Image
import os
import itertools

from dataset import GymDataset
from game import register_custom_frozen_lake, generate_frozen_lake_env, CUSTOM_FROZEN_LAKE_ID
from model.jepa import CNNEncoder, Predictor, JEPA, CNNDecoder

def torch_img_to_np_img(t: torch.Tensor) -> np.array:
    img = t.squeeze().detach().numpy()
    return np.transpose(img, (1, 2, 0))

def plot_img_sequence(images):
    """
    Concatenate image tensors horizontally and display them as a single image.
    This preserves sharpness like the original eval.py.
    """
    images_np = [torch_img_to_np_img(img) for img in images]
    full_sequence = np.concatenate(images_np, axis=1)  # side-by-side
    plt.imshow(full_sequence)
    plt.axis('off')
    plt.tight_layout()
    plt.show()


def load_goal_image(path="img/goal.png"):
    img = Image.open(path).resize((64, 64)).convert("RGB")
    tf = Compose([ToTensor()])
    return tf(img).unsqueeze(0)

def is_goal_reached(decoded_img, goal_img, threshold=0.01):
    loss = F.mse_loss(decoded_img, goal_img)
    return loss.item() < threshold, loss.item()

def simulate_action_sequence(jepa, init_img, action_seq):
    device = next(jepa.parameters()).device
    jepa.eval()
    with torch.no_grad():
        state = jepa.encoder(init_img.to(device))
        images = []
        for action in action_seq:
            action = torch.tensor([action], dtype=torch.long).to(device)
            state = jepa.predictor(state, action)
            recon = jepa.debug_decoder(state)
            images.append(recon.squeeze(0).cpu())
    return images

def dream_search(jepa, init_img, goal_img, max_depth=3):
    all_actions = list(itertools.product([0, 1, 2, 3], repeat=max_depth))
    best_score = float('inf')
    best_seq = None
    best_imgs = None
    for seq in all_actions:
        images = simulate_action_sequence(jepa, init_img, seq)
        final_img = images[-1].unsqueeze(0)
        _, score = is_goal_reached(final_img, goal_img)
        if score < best_score:
            best_score = score
            best_seq = seq
            best_imgs = images
    return best_seq, best_imgs, best_score

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    encoder = CNNEncoder(channels=[3, 4, 8, 16])
    predictor = Predictor(encoder_dim=1024)
    decoder = CNNDecoder(channels=[16, 8, 4, 3], embedding_img_size=(8, 8))
    model = JEPA(encoder=encoder, predictor=predictor, debug_decoder=decoder)

    checkpoint_path = "checkpoints/jepa_with_vicreg.ckpt"
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["state_dict"]
    cleaned_state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(cleaned_state_dict, strict=False)
    model.to(device)
    model.eval()

    goal_img = load_goal_image("img/goal.png").to(device)

    dataset = GymDataset(
        partial(generate_frozen_lake_env, env_id=CUSTOM_FROZEN_LAKE_ID, map_size=4),
        initialize_f=register_custom_frozen_lake,
        transforms=Compose([Resize(64)])
    )

    for x, _, _, _, _ in dataset:
        init_img = x.unsqueeze(0)
        print("Planning in dream...")
        best_seq, images, score = dream_search(model, init_img, goal_img, max_depth=3)
        print("Best sequence:", best_seq)
        print("Goal distance:", score)

        print("Displaying dreamed sequence...")
        plot_img_sequence(images)
        break

if __name__ == "__main__":
    main()
