# ML for OPVs

## Structure

Featurize your dataset for specific feature and save a pkl file in the `data/` folder using
```bash
python featurize.py --dataset=min --feature=mordred
```

Optimize and train your model using `optimize.py`. The best parameters will be saved in `models/`, and predictions for the splits saved in `.csv`.
```bash
python optimize.py --model=ngboost --dataset=min --feature=mordred --n_trails=200
```

To just train your model, use `run_model.py`. The results are saved in `trained_results/`.
```bash
python run_model.py --model=ngboost --dataset=min --feature=mordred
```

## Accessing graph embeddings

To run graph embeddings, read in pickle files from `trained_results/`. Because the models are trained on the splits, you will need to load the features for each split separately.

```python
import pickle

g_embed = pickle.load(open('graphembed_split0.pkl', 'rb'))
``


