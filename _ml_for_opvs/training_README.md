# README for training ML Models
## How to run sklearn models from command line
### Prior to running:
- Make sure you have read and completed everything in data_README.md
- Make sure in the ml_for_opvs/ml_for_opvs/data/input_representation/\*/KFold has input_train_[0-9].csv and input_val_[0-9].csv

### How to train:
1. From command line, Go to ml_for_opvs/ml_for_opvs/ML_models/sklearn/OPV_Min
2. Enter in command line: `./opv_train_run_auto.sh` to run the default training parameters
3. To modify these parameters, edit the `opv_train_run_auto.sh` file.
4. To understand how to train the models:
    - go to ml_for_opvs/ml_for_opvs/ML_models/sklearn.
    - Enter in command line: `python train.py --help` and a list of arguments will be required to run the training loop.
5. By changing the arguments in `opv_train_run_auto.sh` you can train different models, different features, different targets, etc.
