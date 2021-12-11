from dm_control import suite

from ostrichrl.tasks import cassie
from ostrichrl.tasks import ostrich


suite._DOMAINS['cassie'] = cassie
suite._DOMAINS['ostrich'] = ostrich
