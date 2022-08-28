from rasa_core.channels import HttpInputChannel
from rasa_core.agent import Agent
from rasa_core.interpreter import RasaNLUInterpreter
from rasa_slack_connector import SlackInput

if __name__ == '__main__':
    nlu_interpreter = RasaNLUInterpreter('../../src/nlu/projects/default/current/')

    agent = Agent.load('./models/dialogue/', interpreter = nlu_interpreter)
    
    input_channel = SlackInput('xoxp-420079413285-419934888306-419396078145-63335ffd35b0d291d82c43852aec4c76', 'xoxb-420079413285-419727649876-3iJZbpr186AhsZrj8bIb4ZdX', 'hly2PBY7tOlmqMSsBRGgdbpj', True)	
    agent.handle_channel(HttpInputChannel(8088, '/', input_channel))
