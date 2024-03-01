### Code for running the neural networks

```
python torch-NN.py config.json
```

Configuration file:
```
config = {
    "model_type": 'skip',              # the architecture of the base model. Can be skip for sk-DNN or dnn for regular DNN
    "cell_line": 'HepG2',              # the cell 
    "exon_buffer": "100",              # the exon buffer. Its the characteristic of the data
    "data_dir": "../data/",            # the directory where the data is
    "seed": 42,                        # seed is always 42
    "var_y": "target",                 # the column header for the target distribution
    "activation": 'relu',              # the acitvation function being used
    "width": 30,                       # the width of the network
    "depth": 10,                       # the depth of the network. For sk-DNN it is the numbe of blocks.
    "beta" : 0.001,                    # the weight of the ridge regressions
    "alpha": 0.,                       # the weight of the lasso regression
    "scaled": "none",                  # scaling of the data, normally not done
    "lr_decay_type": "exp",            # learning rate decay function. Can be exp, poly, or const
    "initial_lr": 0.001,               # initial learning rate
    "final_lr": 1e-06,                 # final learning rate
    "decay_steps": 500000,             # decayse steps of the learning rate
    "validation_split": 0.2,           # validation split used for early stopping
    "test_split": 0.2,                 # test split used to test the final model
    "batch_size": 256,                 # batch size for stochastic gradient descent
    "steps_per_epoch": 200,            # gradient descent steps per epoch. Early stopping criterion is checked at the end of an epoch
    "patience": 50,                    # the patience for early stopping in unites of epochs
    "monitor": "val_mse",              # the error monitored for early stopping
    "loss": "mse",                     # the loss function. Usually mse.
    'sample_size': 5000000,            # the total smaple size used for training
    "verbose": 1,                      # verbosity
    "base_directory": "../models/",    # the directory in which the final model will be stored
    "epochs": 2000,                    # the maximum number of epochs
    "model-uuid": "UUID",              # the model UUID to make the run unique
    "annotate_cell_lines": False       # annotation of cell lines in the input data. Keep to False for now.
}
```
