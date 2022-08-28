## Generated Story -339354302624317234
* greet
    - utter_greet
    - utter_help
* intent_transfer{"intent": "transfer"}
    - slot{"intent": "transfer"}
    - action_transfer
    - slot{"requested_slot": "type_file"}
* inform
    - action_transfer
    - slot{"type_file": "Climate"}
    - slot{"requested_slot": "endpoint_input"}
* inform{"endpoint_input": "anl"}
    - slot{"endpoint_input": "anl"}
    - action_transfer
    - slot{"endpoint_input": "anl"}
    - slot{"requested_slot": "endpoint_output"}
* inform{"endpoint_output": "lbl"}
    - slot{"endpoint_output": "lbl"}
    - action_transfer
    - slot{"endpoint_output": "lbl"}
    - export

