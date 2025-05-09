import gymnasium as gym
import numpy as np
import pygame
from typing import Optional
from gymnasium.envs.registration import register

from game import run_game, NO_ACTION, QUIT_ACTION, render_onto_screen

if __name__ == '__main__':
    pygame.init()  # Initialize Pygame

    # Register a custom FrozenLake environment with Gymnasium
    register(
        id='FrozenLakeCustom-v0',  # Unique identifier for the environment
        entry_point='frozen_lake:FrozenLakeEnv'  # Entry point to the environment class (must be implemented elsewhere)
    )

    # Mapping of keyboard inputs to corresponding actions in the environment
    key_to_action = {
        pygame.K_RIGHT: 0,  # Move right
        pygame.K_DOWN: 1,  # Move down
        pygame.K_LEFT: 2,  # Move left
        pygame.K_UP: 3,  # Move up
    }


    # Function to process human player input and return an action for the agent
    def human_agent(_: np.array) -> Optional[int]:
        """
        Handles user input and returns the corresponding action.
        - If the user presses a movement key, return the associated action.
        - If the user clicks the close button, return QUIT_ACTION.
        - If no action is taken, return NO_ACTION.

        Parameters:
        - _: np.array (ignored, as this function does not use observations directly)

        Returns:
        - int: Action corresponding to the key pressed, QUIT_ACTION if the user quits, or NO_ACTION otherwise.
        """
        for event in pygame.event.get():  # Process all Pygame events
            if event.type == pygame.QUIT:  # Check if the user closed the window
                return QUIT_ACTION
            elif event.type == pygame.KEYUP:  # Check if a key was released
                if event.key in key_to_action:  # Check if the key corresponds to a movement action
                    return key_to_action[event.key]
        return NO_ACTION  # Default action when no key is pressed


    # Create an instance of the custom FrozenLake environment
    env = gym.make(
        'FrozenLakeCustom-v0',  # ID of the registered environment
        desc=["SFFF", "FHFH", "FFFH", "HFFF"],  # Grid layout (S = Start, F = Frozen, H = Hole, G = Goal)
        is_slippery=False,  # Make movement deterministic (i.e., no slipping)

        # -=== Task 1.1 ===------------
        render_mode="rgb_array"  # Or minimal rendering mode
        #render_mode="minimal"  # Or minimal rendering mode
    )
    env.reset()  # Reset the environment to start a new episode

    # Create a Pygame window for rendering
    screen = pygame.display.set_mode((400, 400))  # Set window size to 400x400 pixels

    # Main loop to run the game
    for img, action in run_game(env, human_agent):  # Run the game loop
        if isinstance(img, np.ndarray):  # If img is an array (game frame), render it
            render_onto_screen(img, screen)
        else:  # Otherwise, print the message (useful for debugging or text-based outputs)
            print(img)

    pygame.quit()  # Quit Pygame once the game ends
    env.close()  # Close the environment properly to free resources