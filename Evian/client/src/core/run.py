from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging

from rasa_core.agent import Agent
from rasa_core.channels.console import ConsoleInputChannel
from rasa_core.interpreter import RegexInterpreter
from rasa_core.policies.keras_policy import KerasPolicy
from rasa_core.policies.memoization import MemoizationPolicy
from rasa_core.interpreter import RasaNLUInterpreter


def run_online(input_channel, nlu_interpreter):
    agent = Agent.load('./models/dialogue/', interpreter = nlu_interpreter)
    agent.handle_channel(input_channel)
    return agent


if __name__ == '__main__':
    nlu_interpreter = RasaNLUInterpreter('../../src/nlu/projects/default/current/')
    run_online(ConsoleInputChannel(), nlu_interpreter)
