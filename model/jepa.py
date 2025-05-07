from typing import Optional

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.preprocessing import StandardScaler


class CNNEncoder(nn.Module):
    """
    CNN-based encoder that processes an input image (state) into a latent representation.
    Used for feature extraction in self-supervised learning models.
    """
#-=========== Task 2.1 CNN encoder -==================
    def __init__(self, channels: list[int]):
        super().__init__()
        # Define a sequential CNN network with multiple layers
        self.net = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=4, kernel_size=3, stride=1, padding=1),  # → [B, 4, 64, 64]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),                                                  # → [B, 4, 32, 32]

            nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3, stride=1, padding=1),  # → [B, 8, 32, 32]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),                                                  # → [B, 8, 16, 16]

            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1), # → [B, 16, 16, 16]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),                                                  # → [B, 16, 8, 8]

            nn.Flatten(),  # Final output shape: [B, 16*8*8] = [B, 1024]
        )


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the CNN encoder.
        :param x: Input image (state)
        :return: Encoded latent representation of the image
        """
        # TODO: Call the net and return the result
        return self.net(x)


class CNNDecoder(nn.Module):
    """
    CNN-based decoder that reconstructs an image from its latent representation.
    Used to verify the quality of the learned latent representations.
    """

    def __init__(self, channels: list[int], embedding_img_size: tuple[int, int]):
        super().__init__()
        # === Task 2.3: CNN decoder to reconstruct 3x64x64 image ===
        self.net = nn.Sequential(
            nn.Unflatten(1, (16, 8, 8)),  # [B, 1024] → [B, 16, 8, 8]
            nn.ConvTranspose2d(16, 8, kernel_size=3, stride=2, padding=1, output_padding=1),  # → [B, 8, 16, 16]
            nn.ReLU(),
            nn.ConvTranspose2d(8, 4, kernel_size=3, stride=2, padding=1, output_padding=1),   # → [B, 4, 32, 32]
            nn.ReLU(),
            nn.ConvTranspose2d(4, 3, kernel_size=3, stride=2, padding=1, output_padding=1),   # → [B, 3, 64, 64]
            nn.Sigmoid()  # Normalize to [0, 1] range for images
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # === Task 2.3: Forward pass of decoder ===
        return self.net(x)



class ActionEncoder(nn.Module):
    """
    Encodes discrete actions into latent representations using an embedding layer.
    """

    def __init__(self, num_actions: int = 4, out_dim: int = 4):
        super().__init__()
        self.encoder = nn.Embedding(num_actions, out_dim)  # Maps actions to latent vectors

    def forward(self, a: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for action encoding.
        :param a: Discrete action input (integer index)
        :return: Latent representation of the action
        """
        return self.encoder(a)


class Predictor(nn.Module):
    """
    Predicts the latent representation of the next state given the current state and action.
    """

    def __init__(self, encoder_dim: int):
        super().__init__()
        self.action_encoder = ActionEncoder(out_dim=32)  # Encodes the action into a latent space
        input_dim = encoder_dim + self.action_encoder.encoder.embedding_dim  # Define input dimension

        # === Task 2.2: Fully connected layers for prediction ===
        self.net = nn.Sequential(
            nn.Linear(input_dim, encoder_dim * 2),  # [s_x + s_a] → 2 * s_x
            nn.ReLU(),
            nn.Linear(encoder_dim * 2, encoder_dim),  # → s_x
        )

    def forward(self, s_x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the predictor.
        :param s_x: Latent representation of input state
        :param a: Discrete action input
        :return: Predicted latent representation of next state
        """
        s_a = self.action_encoder(a)                            # Embed the action
        predictor_input = torch.cat([s_x, s_a], dim=1)          # Concatenate state + action
        predictor_output = self.net(predictor_input)            # Feed through MLP
        return s_x + predictor_output                           # Skip connection



class JEPA(L.LightningModule):
    """
    Joint Embedding Predictive Architecture (JEPA) for self-supervised learning.
    """

    def __init__(
            self,
            encoder: CNNEncoder,
            predictor: Predictor,
            debug_decoder: Optional[CNNDecoder] = None
    ):
        super().__init__()
        self.encoder = encoder  # Encoder for input state
        self.predictor = predictor  # Predicts future state embedding
        self.debug_decoder = debug_decoder  # Optional decoder for visualization
        self.embeddings = []  # Store embeddings for analysis, deactivate by setting it to self.embeddings = None

        # Training hyperparameters
        self.learning_rate = 3e-4
        self.sim_coeff = 25  # Coefficient for similarity loss
        self.std_coeff = 25  # Coefficient for standard deviation regularization
        self.cov_coeff = 1  # Coefficient for covariance regularization


        # Task 3.1(True) and 3.2 (False)
        self.use_vicreg_loss = False

    def vicreg(self, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Computes VICReg loss components: standard deviation loss and covariance loss.
        Standard deviation loss ensures that the variance of embeddings remains high,
        while covariance loss encourages decorrelation between feature dimensions.

        Taken from https://github.com/facebookresearch/vicreg/blob/main/main_vicreg.py
        """

        def off_diagonal(x):
            """
            Extracts off-diagonal elements from a square matrix.
            These elements represent pairwise feature covariances (excluding variances).
            """
            n, m = x.shape
            assert n == m  # Ensure input is a square matrix
            return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

        # Compute standard deviation loss to encourage variance in representations
        std_loss = torch.mean(F.relu(1 - torch.sqrt(x.var(dim=0) + 1e-4))) / 2 + \
                   torch.mean(F.relu(1 - torch.sqrt(y.var(dim=0) + 1e-4))) / 2

        # Compute covariance loss to reduce feature redundancy
        cov_loss = off_diagonal((x.T @ x) / (x.shape[0] - 1)).pow_(2).sum() + \
                   off_diagonal((y.T @ y) / (y.shape[0] - 1)).pow_(2).sum()

        return std_loss, cov_loss
    
    # Task 2.4 -==================
    def training_step(self, batch: torch.Tensor, batch_idx: int):
        """
        Performs a single training step on a batch of data.
        Includes forward passes, loss computation, and logging.
        """
        # === Task 2.4: Unpack batch ===
        x, y, a, x_minimal, y_minimal = batch

        # === Task 2.4: Encode current state ===
        s_x = self.encoder(x)

        # === Task 2.4: Predict next state from s_x and action ===
        s_y_pred = self.predictor(s_x, a)

        # === Task 2.4: Encode next state (ground truth) ===
        s_y = self.encoder(y)

        # === Task 2.4: Compute MSE loss between prediction and ground truth ===
        loss = F.mse_loss(s_y_pred, s_y)


        # Logging dictionary
        log_dict = {
            'diff_sy_sx': loss.item(),
        }

        if self.use_vicreg_loss:
            # Compute VICReg loss components
            std_loss, cov_loss = self.vicreg(s_x, s_y)

            # Compute total loss with weighted contributions
            opt_loss = self.sim_coeff * loss + self.std_coeff * std_loss + self.cov_coeff * cov_loss
            log_dict['std_loss'] = std_loss.item()
            log_dict['cov_loss'] = cov_loss.item()
        else:
            opt_loss = loss

        # Logging dictionary
        log_dict['train_loss'] = opt_loss.item()

        # Store embeddings for visualization if needed
        if self.embeddings is not None:
            self.embeddings.append(s_x)

        # If debug mode is active, run additional decoder step
        if self.debug_decoder is not None:
            opt_loss = self.step_decoder(batch_idx, log_dict, opt_loss, s_x, s_y, s_y_pred, x, x_minimal, y, y_minimal)

        # Log metrics
        self.log_dict(log_dict)

        return opt_loss

    def step_decoder(self, batch_idx: int, log_dict: dict[str, int], opt_loss: torch.Tensor, s_x: torch.Tensor,
                     s_y: torch.Tensor, s_y_pred: torch.Tensor, x: torch.Tensor, x_minimal: torch.Tensor, y,
                     y_minimal: torch.Tensor):
        """ Runs an additional decoder step to reconstruct inputs and log results. """
        # Decode latent representations

        # === Task 2.4: Reconstruct images from latent embeddings ===
        x_rec = self.debug_decoder(s_x.detach())
        y_rec = self.debug_decoder(s_y.detach())
        y_pred_rec = self.debug_decoder(s_y_pred.detach())


        # Compute reconstruction loss
        reconstruction_loss = torch.nn.functional.mse_loss(x_rec, x_minimal)
        opt_loss += reconstruction_loss  # Add reconstruction loss to total optimization loss

        # Log reconstruction loss
        log_dict['reconstruction_loss'] = reconstruction_loss

        # Log reconstructed images for visualization
        self.log_decoder_reconstructions(batch_idx, x, x_minimal, x_rec, y, y_minimal, y_pred_rec, y_rec)

        return opt_loss

    def log_decoder_reconstructions(self, batch_idx: int, x: torch.Tensor, x_minimal: torch.Tensor, x_rec: torch.Tensor,
                                    y: torch.Tensor, y_minimal: torch.Tensor, y_pred_rec: torch.Tensor,
                                    y_rec: torch.Tensor) -> None:
        """ Logs reconstructed images to TensorBoard for visualization every 100 steps. """
        if batch_idx % 100 == 0:
            xx = torch.cat((
                x[0, :],
                x_minimal[0, :],
                x_rec[0, :],
                torch.zeros_like(x[0, :]),
            ), 1)

            yy = torch.cat((
                y[0, :],
                y_minimal[0, :],
                y_rec[0, :],
                y_pred_rec[0, :],
            ), 1)

            xxyy = torch.cat((xx, yy), 2)  # Concatenate x and y horizontally

            self.logger.experiment.add_images('X - Y ; IN | TAR | REC | REC (PRED)', xxyy.unsqueeze(0),
                                              global_step=self.global_step)

    def configure_optimizers(self):
        """ Configures the Adam optimizer with a predefined learning rate. """
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
