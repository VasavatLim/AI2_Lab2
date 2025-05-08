from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.transforms import Compose, Resize

from dataset import GymDataset
from game import register_custom_frozen_lake, generate_frozen_lake_env, CUSTOM_FROZEN_LAKE_ID
from model.jepa import CNNEncoder, Predictor, JEPA, CNNDecoder


def torch_img_to_np_img(t: torch.Tensor) -> np.array:
    """
    Convert a PyTorch tensor image to a NumPy array image.
    The function assumes the tensor is in (C, H, W) format and removes batch dimensions if necessary.
    """
    img = t.squeeze().detach().numpy()  # Remove batch dimension and convert to NumPy
    return np.permute_dims(img, (1, 2, 0))  # Rearrange dimensions to (H, W, C) for visualization


def plot_img(img: np.array) -> None:
    """
    Display an image using Matplotlib without axis labels.
    """
    plt.imshow(img)
    plt.axis('off')  # Remove axis for better visualization
    plt.show()  # Show the image


def dreaming():
    """
    Simulates "dreaming" by predicting future states using a trained JEPA model.
    The function loads a dataset, initializes a JEPA model, and visualizes predicted states.
    """

    # Load the dataset with custom Frozen Lake environment
    dataset = GymDataset(
        partial(generate_frozen_lake_env, env_id=CUSTOM_FROZEN_LAKE_ID),
        initialize_f=register_custom_frozen_lake,
        transforms=Compose([Resize(64)])  # Resize images to 64x64
    )

    # TODO: Instantiate the Encoder, Decoder, and Predictor
    encoder = ...
    predictor = ...
    decoder = ...

    # TODO: Create a JEPA Model where you provide the Encoder, Decoder, and Predictor as arguments
    model = ...

    # TODO: Define how many steps you want to "dream" about the future
    n_steps = ...

    # Iterate over dataset samples
    for x, _, _, _, _ in dataset:
        img = torch_img_to_np_img(x)  # Convert input image to NumPy for visualization
        s_x = model.encoder(x.unsqueeze(0))  # Encode the input image to latent space

        # Generate and visualize future states ("dreaming")
        for i in range(n_steps):  # Predict n steps into the future

            # TODO: Feed s_x through the decoder
            decoded_image = ...

            img = np.concatenate((
                img,
                torch_img_to_np_img(decoded_image)  # Decode and visualize the predicted state
            ), axis=1)

            # TODO: Define an action (torch.Tensor of Shape [1])
            a = ...

            # TODO: Feed s_x and the action through the predictor
            s_y_pred = ...

            # TODO: Set s_x = the predicted action
            s_x = ...

        plot_img(img)  # Display concatenated image with predicted states


if __name__ == "__main__":
    dreaming()
