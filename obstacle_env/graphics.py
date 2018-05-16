from __future__ import print_function, division

import datetime
import shutil

import numpy as np
import pygame
import os

from obstacle_env.scene import PolarGrid


class EnvViewer(object):
    """
        A viewer to render an environment.
    """
    SCREEN_WIDTH = 400
    SCREEN_HEIGHT = 400

    def __init__(self, env, record_video=True):
        self.env = env
        self.record_video = record_video

        pygame.init()
        pygame.display.set_caption("Obstacle-env")
        panel_size = (self.SCREEN_WIDTH, self.SCREEN_HEIGHT)
        self.screen = pygame.display.set_mode([self.SCREEN_WIDTH, self.SCREEN_HEIGHT])
        self.sim_surface = SimulationSurface(panel_size, 0, pygame.Surface(panel_size))
        self.clock = pygame.time.Clock()

    def handle_events(self):
        """
            Handle pygame events by forwarding them to the display and environment vehicle.
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.env.close()
            self.sim_surface.handle_event(event)
            if self.env.dynamics:
                DynamicsGraphics.handle_event(self.env.dynamics, event)

    def display(self):
        """
            Display the scene on a pygame window.
        """
        self.sim_surface.move_display_window_to(self.window_position())
        Scene2dGraphics.display(self.env.scene, self.sim_surface)
        DynamicsGraphics.display(self.env.dynamics, self.sim_surface)
        grid = PolarGrid(self.env.scene)
        grid.trace(self.env.dynamics.position)
        DynamicsGraphics.display_grid(grid, self.sim_surface)
        self.screen.blit(self.sim_surface, (0, 0))
        self.clock.tick(self.env.SIMULATION_FREQUENCY)
        pygame.display.flip()

    def window_position(self):
        """
        :return: the world position of the center of the displayed window.
        """
        if self.env.dynamics:
            return self.env.dynamics.position
        else:
            return np.array([0, 0])

    def close(self):
        """
            Close the pygame window.
        """
        pygame.quit()


class SimulationSurface(pygame.Surface):
    """
           A pygame Surface implementing a local coordinate system so that we can move and zoom in the displayed area.
    """
    BLACK = (0, 0, 0)
    GREY = (100, 100, 100)
    GREEN = (50, 200, 0)
    YELLOW = (200, 200, 0)
    WHITE = (255, 255, 255)
    SCALING_FACTOR = 1.3
    MOVING_FACTOR = 0.1

    def __init__(self, size, flags, surf):
        """
            New window surface.
        """
        super(SimulationSurface, self).__init__(size, flags, surf)
        self.origin = np.array([0, 0])
        self.scaling = 15.0
        self.centering_position = 0.5

    def pix(self, length):
        """
            Convert a distance [m] to pixels [px].

        :param length: the input distance [m]
        :return: the corresponding size [px]
        """
        return int(length * self.scaling)

    def pos2pix(self, x, y):
        """
            Convert two world coordinates [m] into a position in the surface [px]

        :param x: x world coordinate [m]
        :param y: y world coordinate [m]
        :return: the coordinates of the corresponding pixel [px]
        """
        return self.pix(x - self.origin[0, 0]), self.pix(-y + self.origin[1, 0])

    def vec2pix(self, vec):
        """
             Convert a world position [m] into a position in the surface [px].
        :param vec: a world position [m]
        :return: the coordinates of the corresponding pixel [px]
        """
        return self.pix(vec[0]), self.pix(vec[1])

    def rect(self, rect):
        x, y = self.pos2pix(rect[0], rect[1])
        dx, dy = self.vec2pix(rect[2:4])
        return [x, y, dx, dy]

    def move_display_window_to(self, position):
        """
            Set the origin of the displayed area to center on a given world position.
        :param position: a world position [m]
        """
        self.origin = position - np.array(
            [[self.centering_position * self.get_width() / self.scaling], [-self.get_height() / (2 * self.scaling)]])

    def handle_event(self, event):
        """
            Handle pygame events for moving and zooming in the displayed area.

        :param event: a pygame event
        """
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_l:
                self.scaling *= 1 / self.SCALING_FACTOR
            if event.key == pygame.K_o:
                self.scaling *= self.SCALING_FACTOR
            if event.key == pygame.K_m:
                self.centering_position -= self.MOVING_FACTOR
            if event.key == pygame.K_k:
                self.centering_position += self.MOVING_FACTOR


class Scene2dGraphics(object):
    WHITE = (255, 255, 255)
    GREY = (100, 100, 100)

    @staticmethod
    def display(scene, surface):
        pygame.draw.rect(surface, Scene2dGraphics.WHITE, [0, 0, surface.get_width(), surface.get_height()], 0)
        for obstacle in scene.obstacles:
            position = surface.pos2pix(obstacle['position'][0, 0], obstacle['position'][1, 0])
            pygame.draw.circle(surface, Scene2dGraphics.GREY, position, surface.pix(obstacle['radius']), 0)


class DynamicsGraphics(object):
    GREY = (100, 100, 100)
    BLUE = (100, 100, 255)
    COMMAND_LENGTH = 1

    @staticmethod
    def display(dynamics, surface):
        position = surface.pos2pix(dynamics.position[0, 0], dynamics.position[1, 0])
        pygame.draw.circle(surface, Scene2dGraphics.GREY, position, surface.pix(0.2), 1)

        command_position = dynamics.position + dynamics.command * \
                           DynamicsGraphics.COMMAND_LENGTH / dynamics.params['acceleration']
        command_pix = surface.pos2pix(command_position[0, 0], command_position[1, 0])
        pygame.draw.line(surface, DynamicsGraphics.BLUE, position, command_pix)

    @staticmethod
    def display_grid(grid, surface):
        psi = np.repeat(np.arange(-grid.angle/2, 2 * np.pi - grid.angle/2, 2 * np.pi / np.size(grid.grid)), 2)
        psi = np.hstack((psi[1:], [psi[0]]))
        r = np.repeat(np.minimum(grid.grid, grid.MAXIMUM_RANGE), 2)
        # ax.plot(self.origin[0] + r * np.cos(psi), self.origin[1] + r * np.sin(psi), 'k')
        points = [(surface.pos2pix(grid.origin[0] + r[i] * np.cos(psi[i]), grid.origin[1] + r[i] * np.sin(psi[i])))
                  for i in range(np.size(psi))]
        pygame.draw.lines(surface, Scene2dGraphics.GREY, True, points, 1)

    @staticmethod
    def handle_event(dynamics, event):
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_RIGHT:
                dynamics.act("RIGHT")
            if event.key == pygame.K_LEFT:
                dynamics.act("LEFT")
            if event.key == pygame.K_DOWN:
                dynamics.act("DOWN")
            if event.key == pygame.K_UP:
                dynamics.act("UP")
