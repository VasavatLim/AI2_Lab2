

from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.transforms import Compose, Resize, ToTensor

from dataset import GymDataset
from game import register_custom_frozen_lake, generate_frozen_lake_env, CUSTOM_FROZEN_LAKE_ID
from model.jepa import CNNEncoder, Predictor, JEPA, CNNDecoder


def torch_img_to_np_img(t: torch.Tensor) -> np.array:
    """
    Convert a PyTorch tensor image to a NumPy array image.
    The function assumes the tensor is in (C, H, W) format and removes batch dimensions if necessary.
    """
    img = t.squeeze().detach().numpy()  # Remove batch dimension and convert to NumPy
    return np.transpose(img, (1, 2, 0))  # Rearrange dimensions to (H, W, C) for visualization


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
        partial(generate_frozen_lake_env, env_id=CUSTOM_FROZEN_LAKE_ID, map_size=4), #Change map size 4 or 8
        initialize_f=register_custom_frozen_lake,
        transforms=Compose([Resize(64)])
    )


    # TODO: Instantiate the Encoder, Decoder, and Predictor
    encoder = CNNEncoder(channels=[3, 4, 8, 16])
    predictor = Predictor(encoder_dim=1024)
    decoder = CNNDecoder(channels=[16, 8, 4, 3], embedding_img_size=(8, 8))

    # TODO: Create a JEPA Model where you provide the Encoder, Decoder, and Predictor as arguments
    model = JEPA(encoder=encoder, predictor=predictor, debug_decoder=decoder)

    # Load the trained model checkpoint
    # Load the trained model checkpoint from a PyTorch Lightning .ckpt file
    checkpoint_path = "checkpoints/jepa_with_vicreg.ckpt"  # or jepa_no_vicreg.ckpt
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Extract the actual model weights from the Lightning checkpoint
    state_dict = checkpoint["state_dict"]

    # Remove 'model.' prefix if your JEPA class doesn't use it in training
    cleaned_state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}

    model.load_state_dict(cleaned_state_dict, strict=False)
    model.eval()


    # TODO: Define how many steps you want to "dream" about the future
    n_steps = 5
    action_sequence = [1, 1, 2, 2, 1]  # down, down, right, right, down

    # Iterate over dataset samples
    for x, _, _, _, _ in dataset:
        img = torch_img_to_np_img(x)  # Convert input image to NumPy for visualization
        s_x = model.encoder(x.unsqueeze(0))  # Encode the input image to latent space

        output = [x]  # list to store image tensors for visualization

        # Generate and visualize future states ("dreaming")
        for i in range(n_steps):  # Predict n steps into the future

            # TODO: Feed s_x through the decoder
            decoded_image = model.debug_decoder(s_x).squeeze(0)

            output.append(decoded_image)

            # TODO: Define an action (torch.Tensor of Shape [1])
            a = torch.tensor([action_sequence[i]], dtype=torch.long)

            # TODO: Feed s_x and the action through the predictor
            s_y_pred = model.predictor(s_x, a)

            # TODO: Set s_x = the predicted action
            s_x = s_y_pred

        # Convert all images to NumPy and concatenate
        output_np = [torch_img_to_np_img(img) for img in output]
        full_sequence = np.concatenate(output_np, axis=1)

        plot_img(full_sequence)
        break  # only visualize one example


if __name__ == "__main__":
    dreaming()
