from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.transforms import Compose, Resize, ToTensor

from dataset import GymDataset
from game import register_custom_frozen_lake, generate_frozen_lake_env, CUSTOM_FROZEN_LAKE_ID
from model.jepa import CNNEncoder, Predictor, JEPA, CNNDecoder
import heapq
import itertools

# Utility function to convert a torch image tensor (C, H, W) to a NumPy array image (H, W, C)
def torch_img_to_np_img(t: torch.Tensor) -> np.array:
    img = t.squeeze().detach().cpu().numpy()
    return np.transpose(img, (1, 2, 0))

# Display a given image using matplotlib
def plot_img(img: np.array) -> None:
    plt.imshow(img)
    plt.axis('off')
    plt.show()

# Main function that uses the JEPA model to simulate future states (dreaming) and performs a search for an optimal action sequence
# This function implements Task 4.2: using dreaming as a search algorithm
# It searches for the best path (action sequence) up to max_depth using DOWN and RIGHT actions only

def dreaming_path_planner():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load custom Frozen Lake dataset
    dataset = GymDataset(
        partial(generate_frozen_lake_env, env_id=CUSTOM_FROZEN_LAKE_ID),
        initialize_f=register_custom_frozen_lake,
        transforms=Compose([Resize(64)])
    )

    # Instantiate encoder, predictor, decoder, and full JEPA model
    encoder = CNNEncoder(channels=[3, 4, 8, 16]).to(device)
    predictor = Predictor(encoder_dim=1024).to(device)
    decoder = CNNDecoder(channels=[16, 8, 4, 3], embedding_img_size=(8, 8)).to(device)
    model = JEPA(encoder=encoder, predictor=predictor, debug_decoder=decoder).to(device)

    # Load pretrained model checkpoint
    checkpoint_path = "checkpoints/jepa_with_vicreg.ckpt"
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["state_dict"]
    cleaned_state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(cleaned_state_dict, strict=False)
    model.eval()

    actions = [1, 2]  # Only use DOWN and RIGHT actions to bias search toward goal (bottom-right)
    max_depth = 8    # Define max number of actions (search depth)
    visited = set()   # Track visited latent states to avoid loops

    for x, _, _, _, _ in dataset:
        x = x.to(device)
        s_x = model.encoder(x.unsqueeze(0))
        img_start = x.unsqueeze(0)  # Use original (real) image as starting frame

        # Heuristic function (currently returns 0; can be customized to prioritize paths)
        def heuristic(_):
            return 0.0

        # Convert latent tensor to hashable ID for visited tracking
        def state_id(latent):
            return tuple(latent.squeeze().detach().cpu().numpy().round(1))

        # Initialize search frontier using a priority queue (heap)
        counter = itertools.count()
        frontier = [(0.0, next(counter), s_x, [], img_start)]  # (cost, order, latent_state, path, images)
        best_path = None

        # Begin search
        while frontier:
            cost, _, state, path, vis = heapq.heappop(frontier)
            sid = state_id(state)
            if sid in visited:
                continue
            visited.add(sid)

            if len(path) == max_depth:
                best_path = (path, vis)
                break

            for action in actions:
                a = torch.tensor([action], dtype=torch.long, device=device)
                s_next = model.predictor(state, a)             # Predict next latent state
                decoded_img = model.debug_decoder(s_next)      # Decode image from latent

                new_cost = float(cost + 1 + heuristic(s_next))
                heapq.heappush(frontier, (
                    new_cost,
                    next(counter),
                    s_next,
                    path + [action],
                    torch.cat([vis, decoded_img], dim=-1)     # Append decoded image to current visual sequence
                ))

        # If a valid path was found, display the sequence and actions
        if best_path:
            print("Best action sequence:", best_path[0])
            plot_img(torch_img_to_np_img(best_path[1]))
        else:
            print("No path found.")
        break  # Run on only one sample


if __name__ == "__main__":
    dreaming_path_planner()
