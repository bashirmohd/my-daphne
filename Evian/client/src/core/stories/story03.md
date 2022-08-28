## Generated Story 8454196473455982245
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
* inform
    - action_transfer
    - slot{"requested_slot": "endpoint_input"}
* inform{"endpoint_output": "lbl"}
    - slot{"endpoint_output": "lbl"}
    - action_transfer
    - slot{"endpoint_output": "lbl"}
    - slot{"requested_slot": "endpoint_input"}
* inform{"endpoint_input": "anl"}
    - slot{"endpoint_input": "anl"}
    - action_transfer
    - slot{"endpoint_input": "anl"}
    - slot{"requested_slot": "path_output"}
* inform
    - action_transfer
    - slot{"path_output": "~/path/file"}
    - slot{"requested_slot": "path_input"}
* storage/data/file
    - action_transfer
    - slot{"path_input": "/storage/data/file"}
    - export

