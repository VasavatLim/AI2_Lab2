from contextlib import closing
from io import StringIO
from os import path
from typing import List, Optional
import random
import numpy as np

import gymnasium as gym
import pygame
from gymnasium import Env, spaces, utils
from gymnasium.envs.toy_text.utils import categorical_sample
from gymnasium.error import DependencyNotInstalled
from gymnasium.utils import seeding

# Define constants for movement directions (left, down, right, up)
# -==== Task 1.5 ====-
LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3

# Predefined map layouts for the environment
# -==== Task 1.2 =====-
MAPS = {
    "4x4": ["SFFF", "FHFH", "FFFH", "HFFG"],
    "8x8": [
        "SFFFFFFF",
        "FFFFFFFF",
        "FFFHFFFF",
        "FFFFFHFF",
        "FFFHFFFF",
        "FHHFFFHF",
        "FHFFHFHF",
        "FFFHFFFG",
    ],
}


# Depth First Search (DFS) to check if there is a valid path from start to goal
def is_valid(board: List[List[str]], max_size: int) -> bool:
    frontier, discovered = [], set()
    frontier.append((0, 0))
    while frontier:
        r, c = frontier.pop()
        if not (r, c) in discovered:
            discovered.add((r, c))
            directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
            for x, y in directions:
                r_new = r + x
                c_new = c + y
                if r_new < 0 or r_new >= max_size or c_new < 0 or c_new >= max_size:
                    continue
                if board[r_new][c_new] == "G":
                    return True
                if board[r_new][c_new] != "H":
                    frontier.append((r_new, c_new))
    return False


# Function to generate a random map that guarantees a path from start to goal
def generate_random_map(
        size: int = 8, p: float = 0.8, seed: Optional[int] = None
) -> List[str]:
    """
    Generates a random FrozenLake map.

    The map consists of 'S' (starting position), 'F' (frozen surfaces),
    and 'H' (holes). The number of holes is randomly chosen between 1 and 5.
    The function ensures that the starting position 'S' is never a hole.
    """
    grid = np.full((size, size), 'F')  # Fill with 'F' initially

    # Place 'S' at a random position
    s_pos = (random.randint(0, size - 1), random.randint(0, size - 1))
    grid[s_pos] = 'S'

    # Select 1-5 positions for 'H' (excluding 'S')
    num_holes = random.randint(1, 5)
    hole_positions = set()

    while len(hole_positions) < num_holes:
        h_pos = (random.randint(0, size - 1), random.randint(0, size - 1))
        if h_pos != s_pos:
            hole_positions.add(h_pos)

    # Place the holes in the grid
    for h in hole_positions:
        grid[h] = 'H'

    # Convert grid to list of strings for Gym compatibility
    return ["".join(row) for row in grid]



# Define the FrozenLake environment class
class FrozenLakeEnv(Env):
    """
    The FrozenLake environment involves navigating a grid with frozen tiles ('F'),
    holes ('H'), and a goal ('G'). The objective is to move from the start ('S') to
    the goal ('G') while avoiding holes.

    The game mechanics include:
    - A grid world with a start ('S') and goal ('G') position.
    - The player moves randomly, possibly slipping on the ice and moving in unintended directions.
    """

    metadata = {
        "render_modes": ["human", "ansi", "rgb_array", 'minimal'],
        "render_fps": 4,
    }

    def __init__(
            self,
            render_mode: Optional[str] = None,
            desc=None,
            map_name="4x4",
            is_slippery=False,
    ):
        self.render_mode = render_mode
        self.is_slippery = is_slippery

        # Initialize the environment with the provided or default settings
        if desc is None and map_name is None:
            desc = generate_random_map()
        elif desc is None:
            desc = MAPS[map_name]

        # -== Task 1.4 -===========
        self.desc = desc = np.asarray(desc, dtype="c")
        self.nrow, self.ncol = nrow, ncol = desc.shape
        self.reward_range = (0, 1)

        self.setup_env(desc, is_slippery, ncol, nrow, render_mode)

    def setup_env(self, desc, is_slippery, ncol, nrow, render_mode):
        # Number of actions (4 directions: left, down, right, up)
        nA = 4
        # Number of states (total number of grid cells)
        nS = nrow * ncol
        # Initialize the initial state distribution (start position is at [0, 0])
        self.initial_state_distrib = np.array(desc == b"S").astype("float64").ravel()
        self.initial_state_distrib /= self.initial_state_distrib.sum()
        # Transition probability matrix P, where P[s][a] holds the possible transitions from state s for action a
        self.P = {s: {a: [] for a in range(nA)} for s in range(nS)}

        # Function to convert (row, col) to a single state index
        def to_s(row, col):
            return row * ncol + col

        # Function to compute the next position given a current position and an action
        def inc(row, col, a):
            new_row, new_col = row, col
            if a == LEFT:
                new_col = max(col - 1, 0)
            elif a == DOWN:
                new_row = min(row + 1, nrow - 1)
            elif a == RIGHT:
                new_col = min(col + 1, ncol - 1)
            elif a == UP:
                new_row = max(row - 1, 0)
            if not is_hole_at(new_row, new_col):
                return (new_row, new_col)
            else:
                return (row, col)

        # Function to check if there's a hole at the given position
        def is_hole_at(row, col):
            return desc[row, col] == b"H"

        # Update the transition probability matrix P
        def update_probability_matrix(row, col, action):
            new_row, new_col = inc(row, col, action)
            new_state = to_s(new_row, new_col)
            new_letter = desc[new_row, new_col]
            terminated = bytes(new_letter) in b"GH"
            reward = float(new_letter == b"G")
            return new_state, reward, terminated

        # Populate the transition probability matrix for each state and action
        for row in range(nrow):
            for col in range(ncol):
                s = to_s(row, col)
                for a in range(4):
                    li = self.P[s][a]
                    letter = desc[row, col]
                    if letter in b"GH":
                        li.append((1.0, s, 0, True))
                    else:
                        if is_slippery:
                            for b in [(a - 1) % 4, a, (a + 1) % 4]:
                                li.append(
                                    (1.0 / 3.0, *update_probability_matrix(row, col, b))
                                )
                        else:
                            li.append((1.0, *update_probability_matrix(row, col, a)))
        # Define the observation space (discrete space with nS states) and action space (discrete space with 4 actions)
        self.observation_space = spaces.Discrete(nS)
        self.action_space = spaces.Discrete(nA)
        self.render_mode = render_mode
        # Initialize pygame-based renderer for pretty rendering
        self.window_size = (min(64 * ncol, 512), min(64 * nrow, 512))  # Set window size based on grid size
        self.cell_size = (
            self.window_size[0] // self.ncol,
            self.window_size[1] // self.nrow,
        )
        # Initialize renderers for pretty and minimal rendering modes
        self.pretty_renderer = FrozenLakeRenderer(
            ncol, nrow,
            hole_img_path=path.join(path.dirname(__file__), "img/hole.png"),
            cracked_hole_img_path=path.join(path.dirname(__file__), "img/cracked_hole.png"),
            ice_img_path=path.join(path.dirname(__file__), "img/ice.png"),
            elf_images_path=[
                path.join(path.dirname(__file__), "img/elf_left.png"),
                path.join(path.dirname(__file__), "img/elf_down.png"),
                path.join(path.dirname(__file__), "img/elf_right.png"),
                path.join(path.dirname(__file__), "img/elf_up.png"),
            ],
            goal_img_path=path.join(path.dirname(__file__), "img/goal.png"),
            start_img_path=path.join(path.dirname(__file__), "img/stool.png"),
        )
        self.minimal_renderer = FrozenLakeRenderer(
            ncol, nrow,
            hole_img_path=path.join(path.dirname(__file__), "img/minimal/hole.png"),
            cracked_hole_img_path=path.join(path.dirname(__file__), "img/minimal/cracked_hole.png"),
            ice_img_path=path.join(path.dirname(__file__), "img/minimal/ice.png"),
            elf_images_path=[
                path.join(path.dirname(__file__), "img/minimal/elf.png"),
                path.join(path.dirname(__file__), "img/minimal/elf.png"),
                path.join(path.dirname(__file__), "img/minimal/elf.png"),
                path.join(path.dirname(__file__), "img/minimal/elf.png"),
            ],
            goal_img_path=path.join(path.dirname(__file__), "img/minimal/goal.png"),
            start_img_path=path.join(path.dirname(__file__), "img/minimal/stool.png"),
        )

    def step(self, a):
        """
            This function processes one step in the environment.

            - a: The action to take in the current state (s).

            The function retrieves the possible transitions (next states and rewards) from the state-action pair,
            selects a transition based on the probability distribution, and returns the new state, reward,
            whether the episode is done, and a dictionary with additional information (like the probability of the transition).
            """
        transitions = self.P[self.s][a]
        i = categorical_sample([t[0] for t in transitions], self.np_random)
        p, s, r, t = transitions[i]
        self.s = s
        self.lastaction = a

        if self.render_mode == "human":
            self.render()
        # truncation=False as the time limit is handled by the TimeLimit wrapper added during make
        return int(s), r, t, False, {"prob": p}

    def reset(
            self,
            *,
            seed: Optional[int] = None,
            options: Optional[dict] = None,
    ):
        """
            This function resets the environment to an initial state.

            - seed: Optional seed for random number generation.
            - options: Optional dictionary to customize the reset behavior.

            The environment state is randomly initialized based on the distribution of initial states.
            """
        super().reset(seed=seed)


        desc = generate_random_map(self.nrow)
        self.desc = desc = np.asarray(desc, dtype="c")
        self.nrow, self.ncol = nrow, ncol = desc.shape
        self.reward_range = (0, 1)

        self.setup_env(desc, self.is_slippery, ncol, nrow, self.render_mode)

        self.s = categorical_sample(self.initial_state_distrib, self.np_random)
        self.lastaction = None

        if self.render_mode == "human":
            self.render()
        return int(self.s), {"prob": 1}

    def render(self):
        """
        This function renders the environment according to the specified render mode.
        """
        if self.render_mode is None:
            assert self.spec is not None
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym.make("{self.spec.id}", render_mode="rgb_array")'
            )
            return

        match self.render_mode:
            case "ansi":
                return self._render_text()
            case "minimal":
                return self._render_minimal()
            case _:
                return self._render_pretty()

    def _render_pretty(self) -> np.ndarray:
        """
            Renders the environment in a detailed visual format.
            Returns an image (numpy array) of the environment's current state.
            """
        return self.pretty_renderer.render(
            self.desc.tolist(),
            self.s,
            self.lastaction
        )

    def _render_minimal(self) -> np.ndarray:
        """
            Renders the environment in a minimalistic visual format.
            Returns an image (numpy array) of the environment's current state.
            """
        return self.minimal_renderer.render(
            self.desc.tolist(),
            self.s,
            self.lastaction
        )

    @staticmethod
    def _center_small_rect(big_rect, small_dims):
        """
           Helper function to center a smaller rectangle inside a larger rectangle.

           - big_rect: Coordinates (x, y, width, height) of the larger rectangle.
           - small_dims: Dimensions (width, height) of the smaller rectangle.

           Returns the top-left corner (x, y) position for the smaller rectangle to be centered inside the larger one.
           """
        offset_w = (big_rect[2] - small_dims[0]) / 2
        offset_h = (big_rect[3] - small_dims[1]) / 2
        return (
            big_rect[0] + offset_w,
            big_rect[1] + offset_h,
        )

    def _render_text(self):
        """
            Renders the environment as text (ANSI format).
            Outputs a string representation of the environment's grid, highlighting the elf's position.
            """
        desc = self.desc.tolist()
        outfile = StringIO()

        row, col = self.s // self.ncol, self.s % self.ncol
        desc = [[c.decode("utf-8") for c in line] for line in desc]
        desc[row][col] = utils.colorize(desc[row][col], "red", highlight=True)
        if self.lastaction is not None:
            outfile.write(f"  ({['Left', 'Down', 'Right', 'Up'][self.lastaction]})\n")
        else:
            outfile.write("\n")
        outfile.write("\n".join("".join(line) for line in desc) + "\n")

        with closing(outfile):
            return outfile.getvalue()

    def close(self):
        self.pretty_renderer.close()
        self.minimal_renderer.close()


class FrozenLakeRenderer:
    """
        This class is responsible for rendering the FrozenLake environment visually using pygame.

        It takes care of rendering images for holes, cracks, ice, the elf character, the goal, and the start.
        """

    def __init__(
            self,
            ncol: int,
            nrow: int,
            hole_img_path: str,
            cracked_hole_img_path: str,
            ice_img_path: str,
            elf_images_path: list[str],
            goal_img_path: str,
            start_img_path: str,
    ):
        """
                Initializes the renderer with the size of the grid and paths to image assets.

                - ncol, nrow: Number of columns and rows in the grid.
                - hole_img_path, cracked_hole_img_path, ice_img_path, elf_images_path, goal_img_path, start_img_path:
                  Paths to image files representing different elements of the environment.
                """
        self.ncol, self.nrow = ncol, nrow
        self.window_size = (min(64 * ncol, 512), min(64 * nrow, 512))
        self.cell_size = (
            self.window_size[0] // ncol,
            self.window_size[1] // nrow,
        )
        self.window_surface = None
        self.clock = None
        self.hole_img = None
        self.cracked_hole_img = None
        self.ice_img = None
        self.elf_images = None
        self.goal_img = None
        self.start_img = None

        self.hole_img_path = hole_img_path
        self.cracked_hole_img_path = cracked_hole_img_path
        self.ice_img_path = ice_img_path
        self.elf_images_path = elf_images_path
        self.goal_img_path = goal_img_path
        self.start_img_path = start_img_path

    def _init_imgs(self):
        """
                Loads the image files into memory and scales them to fit the grid cells.
                """
        if self.hole_img is None:
            self.hole_img = pygame.transform.scale(
                pygame.image.load(self.hole_img_path), self.cell_size
            )
        if self.cracked_hole_img is None:
            self.cracked_hole_img = pygame.transform.scale(
                pygame.image.load(self.cracked_hole_img_path), self.cell_size
            )
        if self.ice_img is None:
            self.ice_img = pygame.transform.scale(
                pygame.image.load(self.ice_img_path), self.cell_size
            )
        if self.goal_img is None:
            self.goal_img = pygame.transform.scale(
                pygame.image.load(self.goal_img_path), self.cell_size
            )
        if self.start_img is None:
            self.start_img = pygame.transform.scale(
                pygame.image.load(self.start_img_path), self.cell_size
            )
        if self.elf_images is None:
            self.elf_images = [
                pygame.transform.scale(pygame.image.load(f_name), self.cell_size)
                for f_name in self.elf_images_path
            ]

    def _init_window_surface(self):
        """
                Initializes the pygame window surface for rendering the environment.
                """
        try:
            import pygame
        except ImportError as e:
            raise DependencyNotInstalled(
                'pygame is not installed, run pip install "gymnasium[toy-text]"'
            ) from e

        if self.window_surface is None:
            pygame.init()
            self.window_surface = pygame.Surface(self.window_size)

    def render(self, desc, s: int, lastaction):
        """
        Renders the environment by displaying the grid with the elf character in its current position.

        - desc: Description of the grid layout (state).
        - s: The current state (position of the elf).
        - lastaction: The last action taken by the elf.

        Returns the rendered environment as an image (numpy array).
        """
        self._init_window_surface()
        self._init_imgs()

        assert (
                self.window_surface is not None
        ), "Something went wrong with pygame. This should never happen."

        assert isinstance(desc, list), f"desc should be a list or an array, got {desc}"
        for y in range(self.nrow):
            for x in range(self.ncol):
                pos = (x * self.cell_size[0], y * self.cell_size[1])
                rect = (*pos, *self.cell_size)

                self.window_surface.blit(self.ice_img, pos)
                if desc[y][x] == b"H":
                    self.window_surface.blit(self.hole_img, pos)
                elif desc[y][x] == b"G":
                    self.window_surface.blit(self.goal_img, pos)
                elif desc[y][x] == b"S":
                    self.window_surface.blit(self.start_img, pos)

                pygame.draw.rect(self.window_surface, (180, 200, 230), rect, 1)

        # paint the elf
        bot_row, bot_col = s // self.ncol, s % self.ncol
        cell_rect = (bot_col * self.cell_size[0], bot_row * self.cell_size[1])
        last_action = lastaction if lastaction is not None else 1
        elf_img = self.elf_images[last_action]

        if desc[bot_row][bot_col] == b"H":
            self.window_surface.blit(self.cracked_hole_img, cell_rect)
        else:
            self.window_surface.blit(elf_img, cell_rect)

        return np.transpose(
            np.array(pygame.surfarray.pixels3d(self.window_surface)), axes=(1, 0, 2)
        )

    def close(self):
        """
        Closes the pygame display and quits pygame.
        """
        pygame.display.quit()
        pygame.quit()

# Elf and stool from https://franuka.itch.io/rpg-snow-tileset
# All other assets by Mel Tillery http://www.cyaneus.com/