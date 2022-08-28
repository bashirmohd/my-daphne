from flask import Flask, request, jsonify
from flask_restful import Resource, Api
from graphviz import Digraph
import time

app = Flask(__name__)
api = Api(app)

import math as m


stored_intents = []

# Compute the bandwith according the size of the file
#TODO Add Begin and End Time to compute a better bandwith
def minimum_bandwidth(file_size, unit_file_size):#BeginTime #EndTime
        conv = {'b': 1, 'kb': 1024}
        conv['mb'] = 1024 * conv['kb']
        conv['gb'] = 1024 * conv['mb']
        conv['tb'] = 1024 * conv['gb']
        conv['pb'] = 1024 * conv['tb']
        file_size = float(file_size)
        time = 3600
        return (m.ceil((file_size * conv[unit_file_size]) / time))

# Create the second rdf and stores it
def create_graph(multidict, time_server):
        dot = Digraph(comment='RDF2', format='png')
        dot.node('R', '/create/')
        dot.node('time_request', multidict['local_time'])
        dot.node('time_server', time_server)
        dot.node('I', multidict['intent'])
        dot.node('e', 'endpoints')
        dot.node('E', multidict['hasInput'])
        dot.node('F', multidict['hasOutput'])
        dot.node('i', 'file_information')
        dot.node('t', 'time')
        dot.node('p1', multidict['hasInputPath'])
        dot.node('p2', multidict['hasOutputPath'])
        dot.node('exact_time', multidict['hasExactTime'])
        dot.node('minimum_bandwidth', str(minimum_bandwidth(multidict['hasSizeFile'], multidict['hasUnitSizeFile'])) + " B/s")
        dot.node('size', multidict['hasSizeFile'] + multidict['hasUnitSizeFile'])
        dot.node('UI', 'UserInformation')
        dot.node('UIlogin', 'login')
        dot.node('UItoken', 'token')
        ##EDGES
        dot.edge('R', 'I', 'hasIntent')
        dot.edge('R', 'time_request', 'hasLocalTime')
        dot.edge('R', 'time_server', 'hasTimeServer')
        dot.edge('I', 'e', 'hasEndpoints')
        dot.edge('e', 'E', 'hasInput')
        dot.edge('e', 'F', 'hasOutput')
        dot.edge('I', 'i', 'hasFileInformation')
        dot.edge('I', 't', 'hasTimeRequirements')
        dot.edge('i', 'p1', 'hasInputPath')
        dot.edge('i', 'p2', 'hasOutputPath')
        dot.edge('i', 'size', 'hasSizeFile')
        dot.edge('I', 'minimum_bandwidth', 'hasMinimumBandWidth')
        dot.edge('t', 'exact_time', 'hasExactTime')
        dot.edge('R', 'UI', 'hasUserInformation')
        dot.edge('UI', 'UIlogin', 'haslogin')
        dot.edge('UI', 'UItoken', 'hasToken')

        dot.render('./graphs/rdf2.gv', view=True)

# Create an intent
class CreateIntent(Resource):
    def post(self):
        d = request.form
        create_graph(request.form, time.strftime('%A %B, %d %Y %H:%M:%S'))
        #Basic conflicts checking
        if (d['hasExactTime'] == '10pm'):
            stored_intents.append(d)
            return {}, 201
        return jsonify({'time': ['10pm', '11am']})

# List of the stored intents
class IntentList(Resource):
    def get(self):
        return {'intent': stored_intents}, 201

#Intent according the id
class Intent(Resource):
    def get(self, id_intent):
        return {'intent': id_intent}
    def delete(self, id_intent):
        return {'intent': id_intent}
    def modify(self, id_intent):
        return {'intent': id_intent}

#Restful endpoints to provide what are the user endpoints
class Endpoints(Resource):
    def get(self):
        return jsonify({'endpoints' : ['anl', 'lbl']})

api.add_resource(CreateIntent, '/create/')
api.add_resource(IntentList, '/intents/')
api.add_resource(Intent, '/intent/<int:id_intent>')
api.add_resource(Endpoints, '/endpoints/')

if __name__ == '__main__':
    app.run(debug=True, threaded=True)
