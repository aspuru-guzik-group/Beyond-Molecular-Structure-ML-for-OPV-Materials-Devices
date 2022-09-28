# ML for OPVs

## Structure

Featurize your dataset for specific feature and save a pkl file in the `data/` folder using
```bash
python featurizer.py --dataset=min --feature=mordred
```

Optimize and train your model using. The best parameters will be saved in `models/` along with predictions in a `csv` file.
```bash
python train.py --model=ngboost --dataset=min --feature=mordred
```

