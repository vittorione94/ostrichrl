import collections
import os

from dm_control import mujoco
from dm_control.rl import control
from dm_control.suite import base
from dm_control.utils import xml_tools
from lxml import etree
import numpy as np

from ostrichrl.tasks.ostrich import SUITE


class Physics(mujoco.Physics):
    def qpos(self):
        return self.data.qpos.copy()

    def qvel(self):
        qvel = self.data.qvel.copy()
        qvel = np.clip(qvel, -100, 100)
        return qvel

    def muscle_lengths(self):
        return self.data.actuator_length.copy()

    def muscle_velocities(self):
        velocities = self.data.actuator_velocity.copy()
        velocities = np.clip(velocities, -100, 100)
        return velocities

    def muscle_activations(self):
        activations = self.data.act.copy()
        activations = np.clip(activations, -100, 100)
        return activations

    def muscle_forces(self):
        forces = self.data.actuator_force.copy()
        forces /= 1000
        forces = np.clip(forces, -100, 100)
        return forces

    def beak(self):
        return self.named.data.site_xpos['beak'].copy()


class Task(base.Task):
    def __init__(self, random=None):
        super().__init__(random=random)

        self.prev_pos = None
        self.target = np.array([0, 0, 0])
        self.forbidden_sphere_radius = 0.6
        self.allowed_sphere_radius = 0.8
        self.threshold = 5e-2

    def initialize_episode(self, physics):
        if self.prev_pos is None:
            self.prev_pos = np.zeros_like(physics.data.qpos[:])

        # Set initial neck position
        physics.named.data.qpos[:] = self.prev_pos

        while True:
            # Set target position
            phi = self.random.uniform(low=0, high=2 * np.pi)
            costheta = self.random.uniform(low=-1, high=1)
            u = self.random.uniform(low=0, high=1)

            theta = np.arccos(costheta)
            r = self.allowed_sphere_radius * np.cbrt(u)

            x = r * np.sin(theta) * np.cos(phi)
            y = r * np.sin(theta) * np.sin(phi)
            z = r * np.cos(theta)

            self.target = np.array([x, y, z])
            self.target += physics.named.data.site_xpos['allowed_sphere']

            # check if self.target is inside the threshol_sphere
            dist = np.linalg.norm(
                self.target - physics.named.data.site_xpos['forbidden_sphere'])
            if not dist <= self.forbidden_sphere_radius:
                break

    def after_step(self, physics):
        # In env.step() a call to physics.step() is made and that also resets
        # the position of the target site. Specifying this function is also
        # important because there's an issue with visualize reward.
        physics.named.data.site_xpos['target'] = self.target
        return

    def get_reward(self, physics):
        target_site = physics.named.data.site_xpos['target']
        beak_site = physics.named.data.site_xpos['beak']
        self.distance = np.sqrt(((target_site - beak_site) ** 2).sum())
        return -self.distance

    def get_termination(self, physics):
        # The last pose is used to initialize the next episode.
        self.prev_pos = physics.data.qpos.copy()
        if self.distance <= self.threshold:
            return 1

    def get_observation(self, physics):
        obs = collections.OrderedDict()

        obs['qpos'] = physics.qpos()
        obs['qvel'] = physics.qvel()

        obs['muscle_activations'] = physics.muscle_activations()
        obs['muscle_forces'] = physics.muscle_forces()
        obs['muscle_lengths'] = physics.muscle_lengths()
        obs['muscle_velocities'] = physics.muscle_velocities()

        obs['beak'] = physics.beak()
        obs['target'] = self.target.copy()
        obs['target_beak'] = self.target - physics.beak()

        return obs


@SUITE.add('benchmarking')
def foraging(
    environment_kwargs=None, random=None,
):
    task = Task(random=random)

    path = os.path.dirname(__file__)
    path += '/../../assets/models/ostrich/ostrich_neck.xml'
    physics = Physics.from_xml_path(path)
    environment_kwargs = environment_kwargs or {}
    env = control.Environment(
        physics, task, time_limit=25, control_timestep=0.025,
        **environment_kwargs)

    return env
