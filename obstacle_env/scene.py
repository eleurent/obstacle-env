from __future__ import print_function, division
import numpy as np
import math


class Scene1D(object):
    BOUNDS = 10

    def __init__(self):
        self.obstacles = [{'position': np.array([-self.BOUNDS]), 'radius': 1},
                          {'position': np.array([+self.BOUNDS]), 'radius': 1}]


class Scene2D(object):
    BOUNDS = 10

    def __init__(self):
        self.obstacles = \
            [
                {'position': np.array([[self.BOUNDS], [0]]), 'radius': 1},
                {'position': np.array([[2*self.BOUNDS], [2]]), 'radius': 1},
                {'position': np.array([[2*self.BOUNDS], [-2]]), 'radius': 1},
            ]


class PolarGrid:
    MAXIMUM_RANGE = 10

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
            center_angle = self.position_to_angle(obstacle['position'])
            center_distance = np.linalg.norm(obstacle['position'] - origin)
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
        return np.arctan2(position[1, 0] - self.origin[1, 0], position[0, 0] - self.origin[0, 0])

    def position_to_index(self, position):
        return self.angle_to_index(self.position_to_angle(position))

    def angle_to_index(self, angle):
        return int(math.floor(angle / self.angle)) % self.cells

    def index_to_direction(self, index):
        return np.array([[math.cos((index + 0.5) * self.angle)], [math.sin((index + 0.5) * self.angle)]])

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