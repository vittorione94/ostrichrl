import collections
import os

from dm_control import mujoco
from dm_control.rl import control
from dm_control.suite import base
import numpy as np

from ostrichrl.tasks.ostrich import SUITE


class Physics(mujoco.Physics):
    def qpos(self):
        return self.data.qpos.copy()

    def qvel(self):
        return np.clip(self.data.qvel, -100, 100)

    def pelvis_height(self):
        return self.named.data.geom_xpos['pelvis', 'z'].copy()

    def feet_height(self):
        return np.array([self.named.data.xpos['r_pes', 'z'],
                         self.named.data.xpos['l_pes', 'z']])

    def muscle_lengths(self):
        return self.data.actuator_length.copy()

    def muscle_velocities(self):
        return np.clip(self.data.actuator_velocity, -100, 100)

    def muscle_activations(self):
        return np.clip(self.data.act, -100, 100)

    def muscle_forces(self):
        return np.clip(self.data.actuator_force / 1000, -100, 100)


class MoCapTask(base.Task):
    def __init__(
        self, clip, min_episode_steps, pose_rew_weight, rot_rew_weight,
        init_noise_scale, rew_threshold, test, play, random,
    ):
        super().__init__(random=random)
        self._min_episode_steps = min_episode_steps
        self._pose_rew_weight = pose_rew_weight
        self._rot_rew_weight = rot_rew_weight
        self._init_noise_scale = init_noise_scale
        self._rew_threshold = rew_threshold
        self._test = test
        self._play = play
        self._mocap_index = 0
        self._max_mocap_index = 1
        self.initialize_clip(clip)

    def initialize_clip(self, clip):
        path = os.path.dirname(__file__) + '/../../assets/mocap/ostrich/'

        # Joints.
        qpos_path = path + 'qpos/' + clip + '.npy'
        self._mocap_qpos = np.load(qpos_path)

        # Velocities (approximated).
        qvel_path = path + 'qvel/' + clip + '.npy'
        self._mocap_qvel = np.load(qvel_path)

        # Xipos.
        xipos_path = path + 'xipos/' + clip + '.npy'
        self._mocap_xipos = np.load(xipos_path)

        # Ximat.
        ximat_path = path + 'ximat/' + clip + '.npy'
        self._mocap_ximat = np.load(ximat_path)

        self._clip_length = self._mocap_qpos.shape[0]

        self._num_joints = len(self._mocap_qpos[0])
        self._num_bodies = len(self._mocap_xipos[0])

    def initialize_episode(self, physics):
        if self._test:
            self._mocap_index = 0
        else:
            self._mocap_index = self.random.randint(
                self._clip_length - self._min_episode_steps)
        self._max_mocap_index = self._clip_length - 1

        # Joints.
        target_qpos = self._mocap_qpos[self._mocap_index]
        physics.data.qpos[:] = target_qpos
        if not self._test and self._init_noise_scale:
            physics.data.qpos[:] += self.random.normal(
                0, self._init_noise_scale, self._num_joints)

        # Velocities.
        target_qvel = self._mocap_qvel[self._mocap_index]
        physics.data.qvel[:] = target_qvel

    def after_step(self, physics):
        self._mocap_index += 1

        if self._play:
            target_qpos = self._mocap_qpos[self._mocap_index]
            physics.data.qpos[:] = target_qpos
            physics.data.qvel[:] = 0
            physics.forward()

        # Xipos.
        target_xipos = self._mocap_xipos[self._mocap_index]
        xipos = physics.data.xipos[2:]
        xipos_dists = np.linalg.norm(target_xipos - xipos, axis=-1)
        xipos_rew = np.prod(np.exp(-self._pose_rew_weight * xipos_dists))

        # Ximat.
        target_ximat = self._mocap_ximat[self._mocap_index]
        ximat = physics.data.ximat[2:]
        ximat = np.reshape(ximat, (self._num_bodies, 3, 3))
        target_ximat = np.reshape(target_ximat, (self._num_bodies, 3, 3))
        mul = np.matmul(target_ximat, np.transpose(ximat, (0, 2, 1)))
        angles = np.arccos(
            np.clip((np.trace(mul, axis1=1, axis2=2) - 1) / 2, -1, 1))

        ximat_rew = np.prod(np.exp(-self._rot_rew_weight * angles))

        self._reward = xipos_rew * ximat_rew

    def get_observation(self, physics):
        obs = collections.OrderedDict()

        obs['pelvis_height'] = physics.pelvis_height()
        obs['feet_height'] = physics.feet_height()
        obs['qpos'] = physics.qpos()
        obs['qvel'] = physics.qvel()
        obs['muscle_activations'] = physics.muscle_activations()
        obs['muscle_forces'] = physics.muscle_forces()
        obs['muscle_lengths'] = physics.muscle_lengths()
        obs['muscle_velocities'] = physics.muscle_velocities()

        obs['time_left'] = np.array(
            [1.0 - self._mocap_index / self._max_mocap_index])

        return obs

    def get_reward(self, physics):
        return self._reward

    def get_termination(self, physics):
        if self._reward < self._rew_threshold and not self._test:
            return 1

        if self._mocap_index + 1 >= self._clip_length:
            return 1


@SUITE.add('benchmarking')
def mocap_tracking(
    clip='0047', min_episode_steps=20, pose_rew_weight=0.2,
    rot_rew_weight=0.1, environment_kwargs=None, rew_threshold=0.01,
    init_noise_scale=0.02, test=False, play=False, random=None,
):
    task = MoCapTask(
        clip=clip, min_episode_steps=min_episode_steps,
        pose_rew_weight=pose_rew_weight, rot_rew_weight=rot_rew_weight,
        rew_threshold=rew_threshold, init_noise_scale=init_noise_scale,
        test=test, play=play, random=random)

    path = os.path.dirname(__file__)
    path += '/../../assets/models/ostrich/ostrich.xml'
    physics = Physics.from_xml_path(path)
    environment_kwargs = environment_kwargs or {}
    env = control.Environment(
        physics, task, time_limit=10000, control_timestep=0.025,
        **environment_kwargs)

    return env
