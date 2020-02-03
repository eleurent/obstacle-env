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
    SCREEN_WIDTH = 800
    SCREEN_HEIGHT = 800
    SAVE_IMAGES = True

    def __init__(self, env, record_video=True):
        self.env = env
        self.record_video = record_video

        pygame.init()
        pygame.display.set_caption("Obstacle-env")
        panel_size = (self.SCREEN_WIDTH, self.SCREEN_HEIGHT)
        self.screen = pygame.display.set_mode([self.SCREEN_WIDTH, self.SCREEN_HEIGHT])
        self.sim_surface = SimulationSurface(panel_size, 0, pygame.Surface(panel_size))
        self.clock = pygame.time.Clock()

        self.agent_display = None
        self.agent_surface = None
        self.frame = 0

    def set_agent_display(self, agent_display):
        if self.agent_display is None:
            self.agent_display = agent_display
            self.screen = pygame.display.set_mode((self.SCREEN_WIDTH * 2, self.SCREEN_HEIGHT))
            self.agent_surface = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))

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

    def display(self, show_grid=False):
        """
            Display the scene on a pygame window.
        """
        self.sim_surface.move_display_window_to(self.window_position())
        Scene2dGraphics.display(self.env.scene, self.sim_surface)
        DynamicsGraphics.display(self.env.dynamics, self.sim_surface)
        if show_grid:
            self.env.grid.trace(self.env.dynamics.position)
            DynamicsGraphics.display_grid(self.env.grid, self.sim_surface)

        if self.agent_display:
            self.agent_display(self.agent_surface, self.sim_surface)
            if self.SCREEN_WIDTH > self.SCREEN_HEIGHT:
                self.screen.blit(self.agent_surface, (0, self.SCREEN_HEIGHT))
            else:
                self.screen.blit(self.agent_surface, (self.SCREEN_WIDTH, 0))

        self.screen.blit(self.sim_surface, (0, 0))
        self.clock.tick(self.env.config["simulation_frequency"])
        pygame.display.flip()

        if self.SAVE_IMAGES:
            pygame.image.save(self.sim_surface, "out/ObstacleEnv/obstacle-env_{}.png".format(self.frame))
            self.frame += 1

    def get_image(self):
        """
        :return: the rendered image as a rbg array
        """
        data = pygame.surfarray.array3d(self.screen)
        return np.moveaxis(data, 0, 1)

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
        self.scaling = 40.0
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
        return self.pix(x - self.origin[0, 0]), self.pix(y - self.origin[1, 0])

    def vec2pix(self, vec):
        """
             Convert a world position [m] into a position in the surface [px].
        :param vec: a world position [m]
        :return: the coordinates of the corresponding pixel [px]
        """
        return self.pos2pix(vec[0], vec[1])

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
            [[self.centering_position * self.get_width() / self.scaling], [self.get_height() / (2 * self.scaling)]])

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
    GREEN = (50, 150, 50)

    @staticmethod
    def display(scene, surface):
        pygame.draw.rect(surface, Scene2dGraphics.WHITE, [0, 0, surface.get_width(), surface.get_height()], 0)
        for obstacle in scene.obstacles:
            position = surface.pos2pix(obstacle['position'][0, 0], obstacle['position'][1, 0])
            pygame.draw.circle(surface, Scene2dGraphics.GREY, position, surface.pix(obstacle['radius']), 0)
        position = surface.pos2pix(scene.goal['position'][0, 0], scene.goal['position'][1, 0])
        pygame.draw.circle(surface, Scene2dGraphics.GREEN, position, surface.pix(scene.goal['radius']), 0)


class DynamicsGraphics(object):
    GREY = (100, 100, 100)
    BLUE = (100, 100, 255)
    RED = (255, 100, 100)
    COMMAND_LENGTH = 1

    @staticmethod
    def display(dynamics, surface, show_desired_control=False):
        position = surface.pos2pix(dynamics.position[0, 0], dynamics.position[1, 0])
        pygame.draw.circle(surface, Scene2dGraphics.GREY, position, surface.pix(0.2), 1)

        # Actual control
        control_position = dynamics.position + dynamics.control * \
                           DynamicsGraphics.COMMAND_LENGTH / dynamics.params['acceleration']
        control_pix = surface.pos2pix(control_position[0, 0], control_position[1, 0])
        pygame.draw.line(surface, DynamicsGraphics.BLUE, position, control_pix)

        # Desired control
        if show_desired_control:
            desired_control = dynamics.action_to_control(dynamics.desired_action) / dynamics.params['acceleration']
            control_position = (20, surface.get_height()-20)
            control_destination = (control_position[0] + desired_control[0, 0]*10,
                                   control_position[1] + -desired_control[1, 0]*10)
            pygame.draw.circle(surface, DynamicsGraphics.RED, control_position, 4, 1)
            pygame.draw.line(surface, DynamicsGraphics.RED, control_position, control_destination, 2)

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
                dynamics.desired_action = 4
            if event.key == pygame.K_LEFT:
                dynamics.desired_action = 3
            if event.key == pygame.K_DOWN:
                dynamics.desired_action = 2
            if event.key == pygame.K_UP:
                dynamics.desired_action = 1
