import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id='Dproute-v0',
    entry_point='gym_dproute.envs:DprouteEnv',
)

