from dm_control.utils import containers

SUITE = containers.TaggedTasks()

from ostrichrl.tasks.ostrich import foraging  # noqa
from ostrichrl.tasks.ostrich import mocap_tracking  # noqa
from ostrichrl.tasks.ostrich import run  # noqa
from ostrichrl.tasks.ostrich import run_torque  # noqa
