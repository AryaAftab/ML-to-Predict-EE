# ML-to-Predict-EE
Official repository of [**"A Machine Learning Framework for Predicting Entrapment Efficiency in Niosomal Particles"**](https://www.sciencedirect.com/science/article/abs/pii/S0378517322007578).

## Usage
First, you need to make your data into an **n** by **m+1** matrix, where **n** (rows) is the number of data and **m** (columns) is the number of features, and the last column represents the desired output (entrapment efficiency or any feature you want).

#### Prepare Data

```python
import numpy as np
from sklearn.utils import shuffle

data = ... # your data (n, m+1)

data = data.astype(np.float32) # convert all data dtype to float32
data = shuffle(data, random_state=41) # shuffle dataset

X, Y = data_[:,:-1], data_[:,-1] # seperate input and output (n, m+1) -> (n,m), (n,)
Y = np.expand_dims(Y, axis=-1) # add a dimension (n,) -> (n,1)

X = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0) + 1e-8) # iput min-max normalization
Y = Y / 100 # output normalization
```
#### Use ```Trainer``` class
A class called ‍‍‍‍‍‍```Trainer``` has been prepared for training models. Using ```Trainer```, you can easily and without specialized knowledge specify the model, cost function, model hyperparameters, cost function hyperparameters, etc. Then train your model.

```python
from einops import rearrange
from trainer import Trainer

trainer = Trainer(train_data=(X, Y), # your data
                  model_type="linear", # model_type (linear, polynomial, and dnn)
                  loss_type="gaussian", # loss_type(mae, mse, maem, and gaussian)
                  k_folds=5,
                  n_repeats=1, # number of repeat in each fold
                  epochs=1000, # number of training epoch in each fold
                  batch_size=512, # size of training batch
                  experiment_mode="normal", # training mode (normal or sensitivity)
                  epsilon=0.05, # epsilon in maem loss function (if you use other loss_type you don't need it)
                  in_features=X.shape[-1],
                  o_features=Y.shape[-1],
                  n_h_layers=6, # number of hidden layer for dnn (if you use other model_type you don't need it)
                  drop_rate=0.25, # dropout for dnn (if you use other model_type you don't need it)
                  h_units=256, # number of hidden unit for dnn (if you use other model_type you don't need it)
                  activation="relu", # type of activation function for dnn (if you use other model_type you don't need it)
                  degree=10, # degree of polynomial model (if you use other model_type you don't need it)
                  output_strict=True, # Limit the output to 1 or not
)

trainer() # Start training
```

Showing average metrics for K-fold cross-validation
```python
results = rearrange(trainer.Train_Valid_Metric_Results, "f r m o -> (f r o) m")
results.mean(axis=0)[4:] # MAE, RMSE, R2, STD(if loss_type="gaussian") in order
```

## Citation

If you find our code useful for your research, please consider citing:
```bibtex
@article{kashani2022machine,
  title={A Machine Learning Framework for Predicting Entrapment Efficiency in Niosomal Particles},
  author={Kashani-Asadi-Jafari, Fatemeh and Aftab, Arya and Ghaemmaghami, Shahrokh},
  journal={International Journal of Pharmaceutics},
  pages={122203},
  year={2022},
  publisher={Elsevier}
}
```
