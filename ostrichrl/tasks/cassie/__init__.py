from dm_control.utils import containers

SUITE = containers.TaggedTasks()

from ostrichrl.tasks.cassie import mocap_tracking  # noqa
from ostrichrl.tasks.cassie import run  # noqa
