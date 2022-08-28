from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from rasa_core.channels.console import ConsoleInputChannel
from rasa_core.interpreter import RegexInterpreter
from rasa_core.policies.keras_policy import KerasPolicy
from rasa_core.policies.memoization import MemoizationPolicy
from rasa_core.interpreter import RasaNLUInterpreter
from rasa_core.agent import Agent

#Creates an agent to run online the bot
def run_online(input_channel, interpreter,
                          domain_file="domain.yml",
                          training_data_file='stories/'):
    agent = Agent(domain_file,
                  policies=[MemoizationPolicy(max_history=3), KerasPolicy()],
                  interpreter=interpreter)

    agent.train_online(training_data_file,
                       input_channel=input_channel,
                       batch_size=50,
                       epochs=200,
                       max_training_samples=300)

    return agent


if __name__ == '__main__':
    nlu_interpreter = RasaNLUInterpreter('../nlu/projects/default/current/')
    run_online(ConsoleInputChannel(), nlu_interpreter)
