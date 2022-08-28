## Generated Story -2983617671428138758
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
    - slot{"requested_slot": "path_output"}
* inform
    - action_transfer
    - slot{"path_output": "./test/path"}
    - slot{"requested_slot": "path_input"}
* inform
    - action_transfer
    - slot{"path_input": "~/directory/path"}
    - slot{"requested_slot": "size_file"}
* inform{"size_file": "10.4", "unit_size_file": "tb"}
    - slot{"size_file": "10.4"}
    - action_transfer
    - slot{"size_file": "10.4"}
    - export

