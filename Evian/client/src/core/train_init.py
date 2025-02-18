from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals

from rasa_core.agent import Agent
from rasa_core.policies.keras_policy import KerasPolicy
from rasa_core.policies.memoization import MemoizationPolicy

if __name__ == '__main__':
    training_data_file = './stories/'
    model_path = './models/dialogue'
    #We have two max history greet and action of the intent
    agent = Agent('./domain.yml', policies = [MemoizationPolicy(max_history = 2), KerasPolicy()])
    agent.train(training_data_file, epochs = 500, batch_size = 10, validation_split = 0.2)
    agent.persist(model_path)
