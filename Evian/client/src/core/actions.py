from rasa_core.actions import Action
from rasa_core.actions.forms import FormAction
from rasa_core.actions.forms import EntityFormField
from rasa_core.actions.forms import FreeTextFormField
from graphviz import Digraph
from rasa_core.events import AllSlotsReset
import requests
import json
import time

# address of evian server
app_address="http://127.0.0.1:5000"

def json_request(slots):
    data ={}
    data['intent'] = slots['intent']
    data['requirements'] = {}
    data['hasInput'] =  slots['endpoint_input']
    data['hasOutput'] = slots['endpoint_output']
    data['hasInputPath'] = slots['path_input']
    data['hasOutputPath'] =  slots['path_output']
    data['hasSizeFile'] = slots['size_file']
    data['hasUnitSizeFile'] = slots['unit_size_file']
    data['hasExactTime'] = slots['exact_time']
    data['local_time'] = time.strftime('%A %B, %d %Y %H:%M:%S')
    return data

## Create the first rdf graph
def create_and_store_graph(slots):
    if (slots['intent'] == 'transfer'):
        dot = Digraph(comment='Parsed Intent', format='png')

        dot.node('I', slots['intent'])
        dot.node('e', 'endpoints')
        dot.node('E', slots['endpoint_input'])
        dot.node('F', slots['endpoint_output'])
        dot.node('i', 'file_information')
        dot.node('t', 'time')
        dot.node('p1', slots['path_input'])
        dot.node('p2', slots['path_output'])
        dot.node('exact_time', slots['exact_time'])
        dot.node('size', slots['size_file'] + slots['unit_size_file'])
        ##EDGES
        dot.edge('I', 'e', 'hasEndpoints')
        dot.edge('e', 'E', 'hasInput')
        dot.edge('e', 'F', 'hasOutput')
        dot.edge('I', 'i', 'hasFileInformation')
        dot.edge('I', 't', 'hasTimeRequirements')
        dot.edge('i', 'p1', 'hasInputPath')
        dot.edge('i', 'p2', 'hasOutputPath')
        dot.edge('i', 'size', 'hasSizeFile')
        dot.edge('t', 'exact_time', 'hasExactTime')
        dot.render('./graphs/rdf_graph1.gv', view=True)


#Action Transfer
class ActionTransfer(FormAction):

    RANDOMIZE = False
    def name(self):
        return "action_transfer"

    @staticmethod
    def required_fields():
        return [
            EntityFormField("intent", "intent"),
            FreeTextFormField("type_file"),
            EntityFormField("endpoint_input","endpoint_input"),
            EntityFormField("endpoint_output", "endpoint_output"),
            FreeTextFormField("path_output"),
            FreeTextFormField("path_input"),
            EntityFormField("size_file", "size_file"),
            EntityFormField("exact_time", "begin_time"),
        ]

    def submit(self, dispatcher, tracker, domain):
        #Gathering all the entities extracted
        intent_slot = tracker.get_slot('intent')
        endpoint_input_slot = tracker.get_slot('endpoint_input')
        endpoint_output_slot = tracker.get_slot('endpoint_output')
        path_output_slot = tracker.get_slot('path_output')
        path_input_slot = tracker.get_slot('path_input')
        size_file_slot = tracker.get_slot('size_file')
        unit_size_file_slot = tracker.get_slot('unit_size_file')
        exact_time_slot = tracker.get_slot('exact_time')


        response="""You asked for {}, here are the infomations you have provided : 
        endpoint_input : {}
        endpoint_output : {}
        path_output : {}
        path_input : {}
        size_file : {}
        unit_size_file : {}
        exact_time : {}""".format(intent_slot,
        endpoint_input_slot, endpoint_output_slot,
        path_output_slot,
        path_input_slot,
        size_file_slot,
        unit_size_file_slot,
        exact_time_slot)
        dispatcher.utter_message(response)
        dispatcher.utter_message("I will check if i can set this transfer for you")

        # Creation of the first rdf
	#TODO find why graphviz doesn't work
        #create_and_store_graph(tracker.current_slot_values())
        dispatcher.utter_message("API call")

        # sending the request to evian server
        r = requests.post(app_address + "/create/", data = json_request(tracker.current_slot_values()))
        # Handling the response from the server

        # 200 The intent is not achievable and the server wants to negociate with the user
        if (r.status_code == 200):
            print(r.content)
            dispatcher.utter_message("Your intent is not installable. Here are other time options :")
            #Printing of the other time options
            dispatcher.utter_message("1. " + str(r.json()['time'][0]))
            dispatcher.utter_message("2. " +  str(r.json()['time'][1]))
            dispatcher.utter_message("Or if you want to quit, enter quit")
            # Reset of the exact_time slot
            return [SetSlot('exact_time', None)]

        # 201 The intent is achievable 
        if (r.status_code == 201):
            print(r.content)
            dispatcher.utter_message("The network is set for your transfer")
            # Reset of all slots to be ready for another intent
            return [AllSlotsReset()]

        # 409 The intent is not achievable 
        if (r.status_code == 409):
            dispatcher.utter_message("The requested intent could not be completed due to a conflict with the current state of the resource.")
            return [AllSlotsReset()]
        else:
            dispatcher.utter_message("Something goes wrong, please contact the support.")
        dispatcher.utter_message("Status code : " + str(r.status_code))
        return [AllSlotsReset()]

#Action when the user wants to know all his intents
class ActionShow(Action):

    def name(self):
        return "action_show"
    def run(self, dispatcher, tracker, domain):

        dispatcher.utter_message("API call")
        r = requests.get(app_address + "/intents/")
        dispatcher.utter_message(str(r.json()))
        return 
