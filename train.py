from functools import partial

import lightning as L
from lightning import seed_everything
from lightning.pytorch.callbacks import ModelSummary
from torchvision.transforms import Compose, Resize

from dataset import get_dataloader, GymDataset
from game import register_custom_frozen_lake, generate_frozen_lake_env, CUSTOM_FROZEN_LAKE_ID
from model.jepa import CNNEncoder, Predictor, JEPA, CNNDecoder
from lightning.pytorch.loggers import TensorBoardLogger

from model.jepa import CNNEncoder, CNNDecoder, Predictor  # ensure correct import

# Set random seed for reproducibility
seed_everything(0)

 # --=== Task 2.5 Finalize train.py -=====
def train():
    """
    Trains the JEPA (Joint Embedding Predictive Architecture) model on a custom Frozen Lake environment.
    """
    # Define image properties
    n_input_channels = 3  # Number of input channels (e.g., RGB images)
    img_size = 64  # Image resolution
    batch_size = 64 # TODO 32 or 64
    max_epochs = 50  # TODO 10 or 20


    # Initialize the dataset and dataloader
    train_dataloader = get_dataloader(
        GymDataset(
            partial(generate_frozen_lake_env, env_id=CUSTOM_FROZEN_LAKE_ID),
            initialize_f=register_custom_frozen_lake,
            transforms=Compose([Resize(img_size)])
        ),
        batch_size=batch_size,
    )

    # TODO: Instantiate the Encoder, Decoder, and Predictor
    encoder = CNNEncoder(channels=[3, 4, 8, 16])
    predictor = Predictor(encoder_dim=1024)
    decoder = CNNDecoder(channels=[16, 8, 4, 3], embedding_img_size=(8, 8))
    # TODO: Create a JEPA Model where you provide the Encoder, Decoder, and Predictor as arguments
    model = JEPA(encoder=encoder, predictor=predictor, debug_decoder=decoder)


    # Set up the logger for TensorBoard visualization
    logger = TensorBoardLogger("tb_logs", name="jepa")

    # Initialize the trainer with logging and model summary callback
    trainer = L.Trainer(max_epochs=max_epochs, logger=logger, callbacks=[ModelSummary(max_depth=-1)])

    # Train the model
    trainer.fit(model, train_dataloader)

    # Save the trained model checkpoint
    if model.use_vicreg_loss:
        model_save_path = "checkpoints/jepa_with_vicreg.ckpt"

        trainer.save_checkpoint(model_save_path)
    else:
        print("VICReg disabled => do not save checkpoint.")


# Run the training process when the script is executed
if __name__ == "__main__":
    train()
