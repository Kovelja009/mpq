Prerequisites:
- Install necessary packages by running: ```pip install -r requirements.txt```

Running training:
- Adjust parameters in the script: `training_config.py` (**NOTE**: in order to change bitwidths, it's enough to change `bitwidth` parameter in `weight_param` and `activation_param`)
  - currently quantizer has 3 modes: `fixed`, `bypass`, `mpq` 
- run the following command: `python mpq_main.py`
- for tracking training progress run: `tensorboard  --logdir lightning_logs/` 
