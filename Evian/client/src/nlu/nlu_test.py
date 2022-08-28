#
# File to test in a command line interpreter how the natural language interpreter handles sentences
#

from rasa_nlu.model import Interpreter

# where `model_directory points to the folder the model is persisted in
interpreter = Interpreter.load("./projects/default/current/")
var = ""
while 1:
    var = raw_input("Enter your intent: \n")
    if (var == "quit"):
        break
    var = unicode(var, "utf-8")
    print(interpreter.parse(var))
