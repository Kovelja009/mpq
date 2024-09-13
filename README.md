Prerequisites:
- Install necessary packages by running: ```pip install -r requirements.txt```

Running training:
- Adjust parameters in the script: `training_config.json`
  - currently quantizer has 3 modes: `fixed`, `bypass`, `mpq` 
- run the following command: `python mpq_main.py`
- for tracking training progress run: `tensorboard  --logdir lightning_logs/` 
