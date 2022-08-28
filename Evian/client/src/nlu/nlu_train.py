from rasa_nlu.training_data import load_data
from rasa_nlu.config import RasaNLUModelConfig
from rasa_nlu.model import Trainer
from rasa_nlu import config

#Folder where to find the training_data
training_data = load_data('./data/')

trainer = Trainer(config.load("./config/nlu_config.yml"))

trainer.train(training_data)

model_directory = trainer.persist('./projects/', fixed_model_name = 'current')
