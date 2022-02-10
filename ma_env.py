'''
course - Fall 2021
Code: Reza Ahmadzadeh
restructured by: Mohamed Martini
provided environments were wrapped to interface with the trainer and evaluator scripts
'''

import numpy as np
import pygame as pg

# Collision matrix for the small environment
Coll_small = np.array([[0, 0, 1, 2, 2, 1, 2, 2, 1, 2, 2, 1, 3, 0, 2],
                       [0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0],
                       [0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [2, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 2],
                       [1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1],
                       [2, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 2],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
                       [0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
                       [0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0]])

# Collision matrix for the large environment
Coll_large = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 4, 2, 2, 2, 2, 2, 2, 2, 4, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 4, 2, 2, 2, 2, 2, 2, 2, 4, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 4, 2, 2, 2, 2, 2, 2, 2, 4, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
                       [0, 0, 1, 5, 5, 5, 1, 0, 0, 0, 4, 4, 2, 2, 2, 2, 2, 4, 4, 1, 0, 0, 1, 5, 5, 5, 1, 0, 0, 0],
                       [0, 0, 1, 5, 5, 5, 1, 0, 0, 0, 4, 4, 4, 2, 2, 2, 4, 4, 4, 0, 0, 0, 1, 5, 5, 5, 1, 0, 0, 0],
                       [0, 0, 1, 5, 5, 5, 1, 0, 0, 1, 4, 4, 4, 4, 4, 4, 4, 4, 4, 0, 0, 0, 1, 5, 5, 5, 1, 0, 0, 0],
                       [0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 4, 4, 2, 2, 2, 2, 2, 4, 4, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 4, 2, 2, 2, 2, 2, 2, 2, 4, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 4, 2, 2, 2, 2, 2, 2, 2, 4, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 4, 2, 2, 2, 2, 2, 2, 2, 4, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 4, 2, 2, 2, 2, 2, 2, 2, 4, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 4, 2, 2, 2, 2, 2, 2, 2, 4, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
                       [0, 0, 4, 2, 2, 2, 1, 0, 0, 0, 4, 2, 4, 4, 4, 4, 4, 2, 4, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 4, 2, 2, 2, 1, 0, 0, 0, 4, 4, 4, 4, 4, 4, 4, 4, 4, 0, 0, 0, 1, 3, 0, 0, 0, 0, 0, 0],
                       [0, 0, 4, 2, 2, 2, 1, 0, 0, 1, 4, 2, 4, 4, 4, 4, 4, 2, 4, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 4, 2, 2, 2, 2, 2, 2, 2, 4, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 4, 2, 2, 2, 2, 2, 2, 2, 4, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 4, 2, 2, 2, 2, 2, 2, 2, 4, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0, 0, 0, 0, 0, 0, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
                       [0, 0, 1, 5, 5, 5, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 5, 5, 5, 1, 0, 0, 0],
                       [0, 0, 1, 5, 5, 5, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 5, 5, 5, 1, 0, 0, 0],
                       [0, 0, 1, 5, 5, 5, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 5, 5, 5, 1, 0, 0, 0],
                       [0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
                       [4, 4, 4, 4, 4, 4, 4, 4, 4, 1, 4, 4, 4, 4, 4, 4, 4, 4, 4, 1, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
                       [2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
                       [2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]])


bg_color = pg.Color(0, 0, 0)
wall_color = pg.Color(140, 140, 140)
goal_color = pg.Color(0, 100, 0)
hazard_color = pg.Color(255, 20, 20)
agent_color = pg.Color(120, 120, 0)
line_color = pg.Color(128, 128, 128)
bad_color = pg.Color(100, 0, 0)
ice_color = pg.Color(0, 0, 100)

GOAL_S = np.array((0, 12))
GOAL_L = np.array((14, 23))

color_code = {
    0: bg_color,
    1: wall_color,
    2: bad_color,
    3: goal_color,
    4: ice_color,
    5: bg_color,
}

SCALE = 40
FPS = 5


class MaEnv:
    """provided small environment, restructured to be compatbile with the trainer and evaluator scripts"""

    def __init__(self, n_states, n_agents, actions, goal):
        self.n_states = n_states
        self.actions = actions
        self.n_actions = actions.shape[0]
        self.actions_size = 2
        self.goal = goal
        self.n_agents = n_agents

        self.grid, self.nr, self.nc = self.get_grid()
        self.walls = self.get_walls()
        self.reset()

        self.screen = None

    def get_grid(self):
        grid = Coll_small
        nc, nr = grid.shape
        return grid, nr, nc

    def get_walls(self):
        walls = np.vstack(np.where(self.grid == 1)).T
        return walls

    def init_pg(self):
        pg.init()
        self.clock = pg.time.Clock()
        screen = pg.display.set_mode((self.nc * SCALE, self.nr * SCALE))
        screen.fill(bg_color)
        pg.display.set_caption("Mohamed Martini")
        return screen

    def render(self, fps=FPS):
        if self.screen is None:
            self.screen = self.init_pg()

        # handle agent quiting
        elif self.screen is False:
            return

        # look for quit command
        for event in pg.event.get():
            if event.type == pg.QUIT:
                pg.quit()
                self.screen = False
                return False
        self.clock.tick(fps)

        # color screen
        self.draw_grid()

        # draw agents
        for agent_pos in self.s:
            pg.draw.rect(self.screen, agent_color, (agent_pos[1] * SCALE, agent_pos[0] * SCALE, SCALE, SCALE))
        pg.display.flip()
        return True

    def draw_grid(self):
        for i in range(self.grid.shape[0]):
            for j in range(self.grid.shape[1]):
                color = color_code[self.grid[i, j]]
                pg.draw.rect(self.screen, color, (j * SCALE, i * SCALE, SCALE, SCALE))
        # Horizontal lines
        for i in range(self.nr + 1):
            pg.draw.line(self.screen, line_color, (0, i * SCALE), (self.nc * SCALE, i * SCALE), 2)
        # Vertical lines
        for i in range(self.nc + 1):
            pg.draw.line(self.screen, line_color, (i * SCALE, 0), (i * SCALE, self.nr * SCALE), 2)

    def update_s0(self):
        """return a random initial state"""
        self.s = np.zeros((self.n_agents, self.n_states), dtype=int)

        # # pick and update agent position
        avail_r, avail_c = np.where(self.grid == 0)
        idx = np.random.choice(avail_r.shape[0], size=self.n_agents)
        self.s = np.vstack((avail_r[idx], avail_c[idx])).T.reshape((self.n_agents, self.actions_size))
        return self.s

    def step(self, a):
        '''transition function'''
        terminal = np.zeros(self.n_agents, dtype=bool)
        # handle pos
        target_pos = self.s + self.actions[a]
        target_pos = np.clip(target_pos, 0, self.nr - 1)

        # handle wall collisions
        wall_clsns = [i for i in range(target_pos.shape[0]) if any((self.walls[:] == target_pos[i]).all(1))]
        target_pos[wall_clsns] = self.s[wall_clsns]

        # handle terminal
        terminal[np.unique(np.where(target_pos - self.goal == 0)[0])] = True

        # handle reward
        ds = np.linalg.norm(self.s - self.goal, axis=1)
        dsp = np.linalg.norm(target_pos - self.goal, axis=1)
        reward = ds - dsp

        self.s = target_pos
        return self.s, reward, terminal

    def reset(self):
        self.update_s0()
        return self.s


class EnvNew1(MaEnv):
    """provided large environment, restructured to be compatbile with the trainer and evaluator scripts"""

    def __init__(self, n_states, n_agents, actions, goal):
        super(EnvNew1, self).__init__(n_states, n_agents, actions, goal)
        self.get_grid()

    def get_grid(self):
        grid = Coll_large
        nc, nr = grid.shape
        return grid, nr, nc


if __name__ == "__main__":
    # test environment
    RIGHT = [0, 1]
    LEFT = [0, -1]
    UP = [-1, 0]
    DOWN = [1, 0]
    ACTIONS = np.array([UP, DOWN, RIGHT, LEFT])
    n_agents = 20
    # env = MaEnv(2, n_agents, ACTIONS, GOAL_S)
    env = EnvNew1(2, n_agents, ACTIONS, GOAL_L)
    env.reset()
    for i in range(200):
        s, r, _ = env.step(np.random.choice(4, size=n_agents))
        cont = env.render()
        if not cont:
            break
    pg.quit()
