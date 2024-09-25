import json
import pickle
from collections import defaultdict

from pathlib import Path
from typing import NamedTuple

import numpy as np
import numpy.random
import pygame as pg

pg.init()


def choose(arr, samples):
    return [arr[i] for i in np.random.choice(range(len(arr)), samples)]


class COLOURS:
    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)
    RED = (255, 0, 0)
    GREEN = (0, 255, 0)
    BLUE = (0, 0, 255)
    YELLOW = (160, 32, 255)

    @classmethod
    def interpolate(cls, a, b, steps: int):
        val = np.arange(0, 1., 1 / steps).reshape((steps, 1))
        a = np.array(a)
        b = np.array(b)
        return val * b + (1 - val) * a


class KEYS:
    ESCAPE = 41
    SPACE = 44


class PATHS:
    CDIR = Path(__file__).parent
    GRID = CDIR / "grid_full_0.pkl"
    MARKING = CDIR / "marking.pkl"
    MARKING_V2 = CDIR / "marking_2.pkl"
    RIJKS = CDIR / "rijks.jpg"


class Pickler:
    @classmethod
    def save(cls, obj: object, file: Path):
        with open(file, "wb") as f:
            pickle.dump(obj, f)

    @classmethod
    def load(cls, file, default: object):
        if not file.exists():
            cls.save(default, file)

        with open(file, "rb") as f:
            return pickle.load(f)


class Grid(NamedTuple):
    wall: np.ndarray
    exit: np.ndarray


class SimParameters(NamedTuple):
    n_agents: int
    panic: float = 0.5


class CAController:
    def __init__(self, grid: Grid, params: SimParameters):
        assert grid.wall.shape == grid.exit.shape
        assert not (grid.wall & grid.exit).any()
        self.grid = grid
        self.n_agents = params.n_agents
        self.panic = params.panic

        marking = Pickler.load(PATHS.MARKING, -1)
        assert isinstance(marking, np.ndarray)
        self.marking = marking
        self.valid = ~(self.grid.wall | self.grid.exit)

        assert self.valid.sum() > self.n_agents
        self.agents = [(x, y) for x, y in choose(np.argwhere(self.valid), self.n_agents)]

    def reset_agents(self):
        self.agents = [(x, y) for x, y in choose(np.argwhere(self.valid), self.n_agents)]

    def move_agents(self):
        choices = defaultdict(list)
        taken = set(self.agents)
        for agent_i, (x, y) in enumerate(self.agents):
            assert self.valid[x, y]

            surrounding = [(x + i, y + j) for i in range(-1, 2) for j in range(-1, 2)]
            surrounding = [pos for pos in surrounding if not self.grid.wall[pos] and pos not in taken]
            if not surrounding:
                continue

            best = min([self.marking[pos] for pos in surrounding])
            best_surrounding = [pos for pos in surrounding if self.marking[pos] == best]
            choice, = choose(best_surrounding, 1)
            assert isinstance(choice, tuple)
            choices[choice].append(agent_i)

        moving = {}
        for pos, agents in choices.items():
            assert agents
            chosen_agent, = choose(agents, 1)
            assert isinstance(chosen_agent, int)
            moving[chosen_agent] = pos

        new_agents = []
        for i, old_pos in enumerate(self.agents):
            if np.random.uniform(0, 1) <= self.panic:
                x, y = old_pos
            else:
                x, y = moving.get(i, old_pos)

            if self.grid.exit[x, y]:
                continue

            assert self.valid[x, y]
            new_agents.append((x, y))

        self.agents = new_agents

    # def mark(self):
    #     wall, exit_ = self.grid
    #     visited = exit_.copy()
    #
    #     for i in range(100):
    #         up, down, l, r = self.neighbours(visited, False)
    #         frontier = (up | down | l | r) & ~visited & ~wall
    #
    #         if not frontier.any():
    #             return visited & ~exit_
    #         visited |= frontier
    #
    #     raise Exception("reached 100 epochs before running out of frontier")

    def mark_v2(self):
        wall, exit_ = self.grid
        valid = ~(wall | exit_)

        marking = np.ones_like(wall) * 500
        marking[exit_] = 1

        diag = np.array([[1, 0, 1],
                         [0, 0, 0],
                         [1, 0, 1],
                         ])

        neigh = np.array([[0, 1, 0],
                          [1, 1, 1],
                          [0, 1, 0],
                          ])

        for epoch in range(100):
            nm = marking.copy()
            for i, j in np.argwhere(valid):
                conv = marking[i - 1: i + 2, j - 1: j + 2]
                neighs = conv * neigh + diag * 500
                diags = conv * diag + neigh * 500
                nm[i, j] = min(neighs.min() + 1, diags.min() + 1.5, marking[i, j])
                if nm[i, j] < 500:
                    valid[i, j] = False

            if (nm == marking).all():
                return marking

            marking = nm

        raise Exception("failed to converge")

    # @staticmethod
    # def neighbours(arr: np.ndarray, pad_with: object):
    #     xn: np.ndarray = np.pad(arr, 1)
    #     xn[0] = pad_with
    #     xn[-1] = pad_with
    #     xn[:, 0] = pad_with
    #     xn[:, -1] = pad_with
    #
    #     above = xn[2:, 1:-1]
    #     below = xn[:-2, 1:-1]
    #     left_of = xn[1:-1, 2:]
    #     right_of = xn[1:-1, :-2]
    #     return above, below, left_of, right_of


class CellularAutomation:

    def __init__(self, grid: Grid, pixels_per_cell: int = 10, **kwargs):
        w, h = grid.wall.shape
        self.ppc = pixels_per_cell
        self.display = pg.display.set_mode((self.ppc * w, self.ppc * h))
        self.img = pg.image.load(PATHS.RIJKS).convert()

        self.grid = grid
        self.ca = CAController(grid, SimParameters(**kwargs))
        marking = Pickler.load(PATHS.MARKING, -1)
        assert isinstance(marking, np.ndarray)
        self.marking = marking

    def update_display(self):
        self.display.fill(COLOURS.WHITE)
        self.display.blit(self.img, (0, 0))

        heat = list(COLOURS.interpolate(COLOURS.GREEN, COLOURS.RED, 70))
        # heat += list(COLOURS.interpolate(COLOURS.RED, COLOURS.BLUE, 80))
        # heat = heat[::2]

        w, h = self.grid.wall.shape

        for i in range(w):
            size = i * self.ppc
            pg.draw.line(self.display, COLOURS.BLACK, (size, 0), (size, h * self.ppc))

        for i in range(h):
            size = i * self.ppc
            pg.draw.line(self.display, COLOURS.BLACK, (0, size), (w * self.ppc, size))

        for i in range(w):
            for j in range(h):
                if self.grid.wall[i, j]:
                    col = COLOURS.BLACK
                elif self.grid.exit[i, j]:
                    col = COLOURS.YELLOW
                else:
                    col = heat[int(self.marking[i, j])]
                    # col = (0, max(0, 255 - 2 * self.marking[i, j]), 0)

                pg.draw.rect(
                    self.display,
                    col,
                    (1 + i * self.ppc, 1 + j * self.ppc, self.ppc - 1, self.ppc - 1),
                    width=0
                )

        for x, y in self.ca.agents:
            pg.draw.circle(
                surface=self.display,
                color=COLOURS.BLUE,
                center=(1 + (x + .5) * self.ppc, 1 + (y + .5) * self.ppc),
                radius=self.ppc // 2,
            )

    def run(self):
        pg.event.get()
        self.update_display()
        pg.display.update()

        while True:
            pg.event.get()
            pressed = list(pg.key.get_pressed())
            if pressed[KEYS.SPACE]:
                self.ca.move_agents()
                self.update_display()
                pg.display.update()
                if not self.ca.agents:
                    self.ca.reset_agents()


def main():
    """
    - Purple areas are exits
    - Black areas are walls / out of bounds

    - Press space bar to move agents
    - Press Esc to end sim

    - The map and the grading are loaded from pickle files.
    """

    grid = Pickler.load(PATHS.GRID, -1)
    assert isinstance(grid, Grid), "couldn't load"
    CellularAutomation(grid, n_agents=100).run()


if __name__ == '__main__':
    np.random.seed(42)
    main()
