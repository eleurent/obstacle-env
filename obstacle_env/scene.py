from __future__ import print_function, division
import numpy as np
import math


class Scene1D(object):
    BOUNDS = 10

    def __init__(self):
        self.obstacles = [{'position': np.array([-self.BOUNDS]), 'radius': 1},
                          {'position': np.array([+self.BOUNDS]), 'radius': 1}]


class Scene2D(object):
    BOUNDS_X = 50
    BOUNDS_Y = 50

    def __init__(self):
        self.obstacles = []
        self.goal = None
        self.create_random_scene()

    def create_random_scene(self, np_random=np.random):
        self.obstacles = []
        for _ in range(400):
            o = np.zeros((2, 1))
            while np.linalg.norm(o) < 2:
                o = np.array([[self.BOUNDS_X * (np_random.rand() * 2 - 1)],
                              [self.BOUNDS_Y * (np_random.rand() * 2 - 1)]])
            self.obstacles.append({'position': o, 'radius': 1})
        self.goal = {'position': np.array([[self.BOUNDS_X * (np_random.rand() * 2 - 1)],
                                           [self.BOUNDS_Y * (np_random.rand() * 2 - 1)]]),
                     'radius': 1}

    def create_corridor_scene(self):
        for y in np.linspace(-self.BOUNDS_Y, self.BOUNDS_Y, int(self.BOUNDS_Y)+1):
            self.obstacles.append({'position': np.array([[-self.BOUNDS_X], [y]]), 'radius': 1})
        for x in np.linspace(-self.BOUNDS_X, 5*self.BOUNDS_X, int(3*self.BOUNDS_Y)+1):
            self.obstacles.append({'position': np.array([[x], [-self.BOUNDS_Y]]), 'radius': 1})
            self.obstacles.append({'position': np.array([[x], [self.BOUNDS_Y]]), 'radius': 1})
        self.obstacles.extend([
            {'position': np.array([[10], [0]]), 'radius': 1},
            {'position': np.array([[20], [2]]), 'radius': 1},
            {'position': np.array([[20], [-2]]), 'radius': 1},
        ])


class PolarGrid:
    MAXIMUM_RANGE = 12

    def __init__(self, scene, cells=8):
        self.scene = scene
        self.cells = cells
        self.angle = 2 * math.pi / self.cells
        self.grid = np.ones((self.cells, 1)) * float('inf')
        self.origin = None

    def trace(self, origin):
        self.origin = origin
        self.grid = np.ones((self.cells, 1)) * float('inf')

        for obstacle in self.scene.obstacles:
            center_distance = np.linalg.norm(obstacle['position'] - origin)
            if center_distance > self.MAXIMUM_RANGE:
                continue
            center_angle = self.position_to_angle(obstacle['position'])
            half_angle = math.acos(math.sqrt(max(1-(obstacle['radius']/center_distance)**2, 0)))
            center_index = self.angle_to_index(center_angle)
            self.grid[center_index] = min(self.grid[center_index], center_distance - obstacle['radius'])
            start, end = self.angle_to_index(center_angle - half_angle), self.angle_to_index(center_angle + half_angle)
            if start < end:
                indexes = np.arange(start, end+1)
            else:
                indexes = np.hstack([np.arange(start, self.cells), np.arange(0, end + 1)])

            for index in indexes:
                direction = self.index_to_direction(index)
                distance = self.distance_to_circle(obstacle['position'] - origin, obstacle['radius'], direction)
                self.grid[index] = min(self.grid[index], distance)
        return self.grid

    def position_to_angle(self, position):
        return np.arctan2(position[1, 0] - self.origin[1, 0], position[0, 0] - self.origin[0, 0]) + self.angle/2

    def position_to_index(self, position):
        return self.angle_to_index(self.position_to_angle(position))

    def angle_to_index(self, angle):
        return int(math.floor(angle / self.angle)) % self.cells

    def index_to_direction(self, index):
        return np.array([[math.cos(index * self.angle)], [math.sin(index * self.angle)]])

    @staticmethod
    def distance_to_circle(center, radius, direction):
        scaling = radius * np.ones((2, 1))
        a = np.linalg.norm(direction / scaling) ** 2
        b = -2 * np.dot(np.transpose(center), direction / np.square(scaling))
        c = np.linalg.norm(center / scaling) ** 2 - 1
        root_inf, root_sup = PolarGrid.solve_trinom(a, b, c)
        if root_inf and root_inf > 0:
            distance = root_inf
        elif root_sup and root_sup > 0:
            distance = 0
        else:
            distance = float('inf')
        return distance

    @staticmethod
    def solve_trinom(a, b, c):
        delta = b ** 2 - 4 * a * c
        if delta >= 0:
            return (-b - np.sqrt(delta)) / (2 * a), (-b + np.sqrt(delta)) / (2 * a)
        else:
            return None, None


def test():
    scene = Scene2D()
    grid = PolarGrid(scene)
    grid.trace(np.array([[0], [0]]))
    print(np.transpose(grid.grid))


# if __name__ == 'main':
#     test()