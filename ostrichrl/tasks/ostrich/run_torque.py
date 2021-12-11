import collections
import os

from dm_control import mujoco
from dm_control.rl import control
from dm_control.suite import base
import numpy as np

from ostrichrl.tasks.ostrich import SUITE


class Physics(mujoco.Physics):
    def qpos_without_x(self):
        return self.data.qpos.copy()[1:]

    def qvel(self):
        qvel = self.data.qvel.copy()
        qvel = np.clip(qvel, -100, 100)
        return qvel

    def pelvis_height(self):
        return self.named.data.geom_xpos['pelvis', 'z'].copy()

    def feet_height(self):
        return np.array([self.named.data.xpos['r_pes', 'z'],
                         self.named.data.xpos['l_pes', 'z']])

    def torso_angle(self):
        return self.data.qpos.copy()[4]

    def horizontal_velocity(self):
        return self.named.data.sensordata['torso_subtreelinvel'].copy()[0]


class RunTorque(base.Task):
    def initialize_episode(self, physics):
        limits = physics.data.model.jnt_range[6:]
        physics.data.qpos[6:] = self.random.uniform(
            low=limits[:, 0] / 5, high=limits[:, 1] / 5)
        physics.data.qvel[:] = 0

    def after_step(self, physics):
        return

    def get_observation(self, physics):
        obs = collections.OrderedDict()

        obs['pelvis_height'] = physics.pelvis_height()
        obs['feet_height'] = physics.feet_height()
        obs['qpos'] = physics.qpos_without_x()
        obs['qvel'] = physics.qvel()

        obs['horizontal_velocity'] = physics.horizontal_velocity()

        return obs

    def get_reward(self, physics):
        return physics.horizontal_velocity()

    def get_termination(self, physics):
        if physics.pelvis_height() < 0.6:
            return 1
        if physics.torso_angle() < -0.8 or physics.torso_angle() > 0.8:
            return 1


@SUITE.add('benchmarking')
def run_torque(environment_kwargs=None, random=None):
    task = RunTorque(random=random)

    path = os.path.dirname(__file__)
    path += '/../../assets/models/ostrich/ostrich_legs_torque.xml'
    physics = Physics.from_xml_path(path)
    environment_kwargs = environment_kwargs or {}
    env = control.Environment(
        physics, task, time_limit=25, control_timestep=0.025,
        **environment_kwargs)

    return env
