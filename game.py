
import random
from typing import Generator, Tuple, Callable, Union

import numpy as np
import pygame
import gymnasium as gym
from gymnasium import register, Env

# Constants for actions
NO_ACTION = None  # No action to be taken (used for cases where no action is required)
QUIT_ACTION = -1  # Action to quit the game


def run_game(env, agent: Callable[[np.array], Union[int, NO_ACTION]]) -> Generator[Tuple[np.ndarray, int], None, None]:
    """
    Runs a game loop for the given environment and agent.

    The function repeatedly:
    1. Renders the environment to obtain the current observation.
    2. Passes the observation to the agent to determine the next action.
    3. Yields the observation and action as a tuple.
    4. Takes a step in the environment with the selected action.
    5. If the action is NO_ACTION, the loop continues to the next iteration without stepping.
    6. If the action is QUIT_ACTION, the loop terminates.
    """
    while True:
        obs = env.render()
        action = agent(obs)
        yield obs, action
        if action is NO_ACTION:
            continue
        if action == QUIT_ACTION:
            break
        _, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break


def render_onto_screen(img, screen: pygame.Surface):
    """
        Renders an image onto the Pygame screen by first converting the image
        from a NumPy array to a Pygame surface and scaling it to fit the screen.

        Args:
        - img: A NumPy array representing the image to be rendered.
        - screen: The Pygame surface onto which the image will be drawn.
        """

    def numpy_to_pygame(x: np.array) -> pygame.Surface:
        """
                Converts a NumPy array to a Pygame surface.

                The NumPy array is assumed to be in the shape (height, width, channels),
                but Pygame expects it in (width, height, channels) format. The function
                also flips the image vertically to match the typical rendering orientation.

                Args:
                - x: A NumPy array representing the image.

                Returns:
                - A Pygame surface representing the same image.
                """
        x = np.transpose(x, (1, 0, 2))
        x = np.flip(x, 0)
        x = pygame.surfarray.make_surface(x)
        return pygame.transform.scale(x, (screen.get_width(), screen.get_height()))

    screen.blit(numpy_to_pygame(img), (0, 0))
    pygame.display.flip()


CUSTOM_FROZEN_LAKE_ID = 'FrozenLakeCustom-v0'


def register_custom_frozen_lake() -> None:
    """
    Registers a custom version of the FrozenLake environment with Gym.

    The function sets up the environment to be used with Gym by registering
    the custom environment ID and linking it to the environment class.
    """
    pygame.init()
    register(
        id=CUSTOM_FROZEN_LAKE_ID,
        entry_point='frozen_lake:FrozenLakeEnv'
    )


def generate_frozen_lake_env(env_id: str, map_size: int = 4) -> Env:
    """
    Generates a custom FrozenLake environment with a random map.

    This function uses a helper function to create a random map where 'S' is the starting
    position, 'H' are holes, and 'F' are frozen surfaces. The function then creates a Gym
    environment using this random map and returns it.

    :param env_id: Id of frozen lake environment
    :param map_size: Size of the map
    :return: Frozen lake environment
    """

    # Create and return the Gym environment using the custom map
    return gym.make(
        env_id,
        is_slippery=False,
        render_mode='rgb_array',
    )