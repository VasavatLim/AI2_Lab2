import os
import random
from typing import Generator, Callable, Any

import numpy as np
import torch
from gymnasium import Env
from gymnasium.core import RenderFrame
from torch import Tensor
from torch.utils.data import IterableDataset


class GymDataset(IterableDataset):
    """
    An iterable dataset that generates an infinite stream of samples from episodes
    using a Gym environment.
    """

    def __init__(
            self,
            env_f: Callable[[], Env],  # Function to create a new environment
            initialize_f: Callable[[], None] = None,  # Initialization function for the dataset
            transforms=None,  # Transformation function for the images
            n_steps=30,  # Maximum number of steps per episode
            n_samples=50_000,  # Number of samples to generate
    ):
        """
        Constructor for the GymDataset.

        :param env_f: Function to create a new environment instance.
        :param initialize_f: Function to initialize the dataset (called at the start of each worker).
        :param transforms: Transformations to apply to the images.
        :param n_steps: Maximum number of steps before recreating the environment.
        :param n_samples: Number of samples per episode.
        """
        self.transforms = transforms if transforms is not None else lambda x: x
        self.env_f = env_f
        self.initialize_f = initialize_f
        self.n_steps = n_steps
        self.n_samples = n_samples
        self.initialized = False
        self.env = None

    def _generate_episode(self, n_steps) -> Generator[tuple[
        RenderFrame | list[RenderFrame] | None, RenderFrame | list[RenderFrame] | None, Any, Any, Any], None, None]:
        """
        Generator that creates an episode from the Gym environment.
        """
        retry_count = 0
        while retry_count < 5:
            try:
                if self.env is None:
                    self.env = self.env_f()
                env = self.env
                env.reset()
                for _ in range(n_steps):
                    img_before = env.render()
                    img_before_minimal = getattr(env.unwrapped, "_render_minimal", lambda: None)()
                    action = random.choice([0, 1, 2, 3])
                    _, _, done, _, _ = env.step(action)
                    img_after = env.render()
                    img_after_minimal = getattr(env.unwrapped, "_render_minimal", lambda: None)()
                    yield img_before, img_after, action, img_before_minimal, img_after_minimal
                env.close()
                return
            except RuntimeError as e:
                retry_count += 1
                print(f"[Worker] Environment crashed due to RuntimeError (attempt {retry_count}): {e}")

            except Exception as e:
                retry_count += 1
                print(f"[Worker] Environment crashed (attempt {retry_count}): {e}")


    def __iter__(self) -> Generator[tuple[Any, Any, Tensor, Any, Any], None, None]:
        """
        Creates an iterator over the dataset episodes.
        """
        if not self.initialized:
            self.initialize_f()
            self.initialized = True

        def to_img_tensor(img: np.array) -> torch.tensor:
            """Converts an image into a normalized tensor format."""
            return torch.tensor(img, dtype=torch.float32).permute(2, 0, 1) / 255.0

        for _ in range(self.n_samples):
            # generate and yield episode
            yield from ((
                self.transforms(to_img_tensor(img_before)),
                self.transforms(to_img_tensor(img_after)),
                torch.tensor(action, dtype=torch.long),
                self.transforms(to_img_tensor(img_before_minimal)),
                self.transforms(to_img_tensor(img_after_minimal)),
            ) for img_before, img_after, action, img_before_minimal, img_after_minimal in
            self._generate_episode(self.n_steps))

    def __len__(self):
        """
        Returns the number of samples. Since the dataset is infinite, a fixed number of samples is given.
        """
        return self.n_samples


class ShuffledIterableDataset(IterableDataset):
    """
    A dataset that takes data from another iterable source and shuffles it randomly.
    """

    def __init__(self, source_dataset, buffer_size):
        self.source_dataset = source_dataset
        self.global_buffer_size = buffer_size

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            num_workers = worker_info.num_workers
            worker_id = worker_info.id
            buffer_size = max(1, self.global_buffer_size // num_workers)
        else:
            worker_id = 0
            buffer_size = self.global_buffer_size

        torch.manual_seed(worker_id + random.randint(0, 10000))  # Zufälliger Seed für mehr Entropie
        buffer = []
        dataset_iter = iter(self.source_dataset)

        try:
            for item in dataset_iter:
                if len(buffer) < buffer_size:
                    buffer.append(item)
                else:
                    idx = torch.randint(0, buffer_size, (1,)).item()
                    yield buffer[idx]
                    buffer[idx] = item

            # Ensure remaining elements are yielded
            random.shuffle(buffer)
            for item in buffer:
                yield item

        except StopIteration:
            pass

    def __len__(self):
        """Returns the number of samples if possible, otherwise raises an error."""
        if hasattr(self.source_dataset, "__len__"):
            return len(self.source_dataset)
        else:
            raise TypeError("IterableDataset has no static length.")



def collate_fn(batch):
    """
    Function that merges individual episodes into a batch.
    """
    imgs_before, imgs_after, actions, imgs_before_minimal, imgs_after_minimal = zip(*batch)
    return torch.stack(imgs_before), torch.stack(imgs_after), torch.tensor(actions), torch.stack(
        imgs_before_minimal), torch.stack(imgs_after_minimal)


def worker_init_fn(worker_id):
    """
    Initializes the environment for each worker to prevent issues due to shared states.
    """
    print(f"[Worker {worker_id}] Initializing environment...")


def get_dataloader(dataset: GymDataset, batch_size=32, shuffle_buffer_size=5000, num_workers=None):
    """
    Creates a DataLoader for the GymDataset.

    :param dataset: The dataset to load.
    :param batch_size: Size of batches.
    :param shuffle_buffer_size: Buffer size for shuffling.
    :param num_workers: Number of worker processes.
    :return: DataLoader instance.
    """
    num_workers = num_workers or os.cpu_count()
    print(f"Using {num_workers} CPU cores")

    return torch.utils.data.DataLoader(
        ShuffledIterableDataset(dataset, shuffle_buffer_size),
        batch_size=batch_size,
        collate_fn=collate_fn,
        num_workers=num_workers,
        worker_init_fn=worker_init_fn,  # Ensures each worker initializes properly
        persistent_workers=num_workers > 0,
        pin_memory=True
    )
