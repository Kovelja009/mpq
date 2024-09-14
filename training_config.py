config = {
    "n_epochs": 4,
    "milestones": [30, 45],
    "quantizer_mode" : "fixed",
    "weight_params" : {
        "bitwidth" : 32,
        "signed" : True,
        "step": {
            "init" : 2**-3,
            "min" : 2**-8,
            "max" : 1
        },
        "qmax":{
            "min" : 2**-8
        }
    },
    "activation_params" : {
        "bitwidth" : 32,
        "signed" : False,
        "step": {
            "init" : 2**-3,
            "min" : 2**-8,
            "max" : 1
        },
        "qmax":{
            "min" : 2**-8
        }
    }
}