import os
from functools import partial

import numpy as np
from sklearn.model_selection import KFold

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import losses

from models import dnn_model, linear_model, polynomial_model
from custom_callbacks import lr_scheduler, ShowProgress, BestModelWeights, Sensitivity, COLOR
from custom_losses import GaussLoss, MAEM
from custom_metrics import RSquare, RootMeanSquaredError, MeanAbsoluteError, STD




# Trainer class
class Trainer:
    def __init__(self,
                 train_data,
                 seperate_test_data=None,
                 model_type="dnn",
                 loss_type="mae",
                 k_folds=5,
                 n_repeats=1,
                 epochs=1000,
                 batch_size=None,
                 save_weights_path="weights",
                 init_weights_name="initial_weights.h5",
                 experiment_mode="normal", # normal or sensitivity
                 **kwargs,
                 ):
        
        self.train_data = train_data
        self.seperate_test_data = seperate_test_data
        self.model_type = model_type
        self.loss_type = loss_type
        self.k_folds = k_folds
        self.n_repeats = n_repeats
        self.epochs = epochs
        self.batch_size = batch_size
        self.save_weights_path = save_weights_path
        self.init_weights_name = init_weights_name
        self.experiment_mode = experiment_mode
        self.extra_arg = kwargs


        os.makedirs(self.save_weights_path, exist_ok=True)

        
        if loss_type == "gaussian":
            self.extra_arg["o_features"] = 2 * self.extra_arg["o_features"]

        self.create_model(**self.extra_arg)
          
    

    def create_model(self, **kwargs):
        if self.model_type == "dnn":
            self.model = dnn_model(**kwargs)
        elif self.model_type == "linear":
            self.model = linear_model(**kwargs)
        elif self.model_type == "polynomial":
            self.model = polynomial_model(**kwargs)
        else:
            raise ValueError('model_type not valid!')


        self.init_weights_path = os.path.join(self.save_weights_path, self.init_weights_name)
        if os.path.exists(self.init_weights_path):
            try:
                self.load_weights(self.model, self.init_weights_path)
            except:
                self.save_weights(self.model, self.init_weights_path)
        else:
            self.save_weights(self.model, self.init_weights_path)
    

    def save_weights(self, model, weights_path):
        model.save_weights(weights_path)

    def load_weights(self, model, saved_weights_path):
        model.load_weights(saved_weights_path)

    
    def train(self, model, train_data, valid_data):
        if self.loss_type == "gaussian":
            optimizer = keras.optimizers.Adam(clipvalue=1.0)
        else:
            optimizer = keras.optimizers.Adam()


        if self.loss_type == "mae":
            loss = losses.MeanAbsoluteError()
        elif self.loss_type == "mse":
            loss = losses.MeanSquaredError()
        elif self.loss_type == "maem":
            loss = MAEM(**self.extra_arg)
        elif self.loss_type == "gaussian":
            loss = GaussLoss()
        else:
            raise ValueError('loss_type not valid!')


        if self.loss_type == "gaussian":
            uncertainty = True
        else:
            uncertainty = False


        METRICS = [MeanAbsoluteError(uncertainty=uncertainty, y_shape=train_data[1].shape[-1], name='MAE'),
                   RootMeanSquaredError(uncertainty=uncertainty, y_shape=train_data[1].shape[-1], name='RMSE'),
                   RSquare(uncertainty=uncertainty, y_shape=train_data[1].shape[-1], name='R2'),
                   ]

        if uncertainty:
            METRICS.append(STD(uncertainty=uncertainty, y_shape=train_data[1].shape[-1], name='STD'))


        model.compile(optimizer=optimizer,
                      loss=loss,
                      metrics=METRICS,
                      )


        partial_lr_scheduler = partial(lr_scheduler, warmup_epochs=self.epochs // 4,
                                       decay_epochs=(self.epochs - self.epochs // 4))

        callbacks = [keras.callbacks.LearningRateScheduler(lr_scheduler, verbose=0),
                     BestModelWeights(metric="val_loss", metric_type="min"),
                     ShowProgress(self.epochs, step_show=10, metric="loss"),
                     ]


        if self.experiment_mode == "sensitivity":
            callbacks.append(Sensitivity(model, self.main_prediction, valid_data))


        if self.batch_size is None:
            self.batch_size = len(train_data[0])


        history = model.fit(train_data[0], train_data[1],
                            epochs=self.epochs,
                            batch_size=self.batch_size,
                            validation_data=valid_data,
                            callbacks=callbacks,
                            verbose=0).history


        best_index = np.argmin(history['val_loss'])


        if uncertainty:
            train_std = history['STD'][best_index]
            val_std = history['val_STD'][best_index]
        else:
            train_std = np.zeros_like(history['MAE'][best_index]) 
            val_std = np.zeros_like(history['MAE'][best_index]) 


        if self.experiment_mode == "sensitivity":
            self.Sensitivity_Results.append(np.array(history["sensitivity_result"]).min(axis=0))


        return np.array([history['MAE'][best_index],
                         history['RMSE'][best_index],
                         history['R2'][best_index],
                         train_std,
                         history['val_MAE'][best_index],
                         history['val_RMSE'][best_index],
                         history['val_R2'][best_index],
                         val_std,
                         ])
    

    def __call__(self):

        self.Train_Valid_Metric_Results = []

        if self.experiment_mode == "sensitivity":
            self.Sensitivity_Results = []

        if self.seperate_test_data:
            self.Test_Metric_Results = []

        

        X, Y = self.train_data
        
        KF = KFold(n_splits=self.k_folds)
        for counter, (train_index, valid_index) in enumerate(KF.split(X)):
            print(COLOR.BOLD + COLOR.BLUE + f"Fold {counter + 1} Started!" + COLOR.END)

            train_x, train_y = X[train_index], Y[train_index]
            valid_x, valid_y = X[valid_index], Y[valid_index]


            best_metrics_repeat = []
            best_test_metrics_repeat = []
            for counter1 in range(self.n_repeats):
                print(COLOR.RED + f"Repeat : {counter1 + 1}" + COLOR.END)


                cloned_model = keras.models.clone_model(self.model)


                mode_weights_path = os.path.join(self.save_weights_path, f"{counter+1}.h5")
                if self.experiment_mode == "sensitivity":
                    try:
                        self.load_weights(cloned_model, mode_weights_path)
                        self.main_prediction = cloned_model.predict(valid_x, verbose=0)
                    except:
                        raise ValueError("You must run trainer one time in normal experiment mode")


                cloned_model.load_weights(self.init_weights_path)
                best_metrics = self.train(cloned_model, (train_x, train_y), (valid_x, valid_y))


                if self.seperate_test_data:
                    test_metrics = cloned_model.evaluate(self.seperate_test_data[0], self.seperate_test_data[1], verbose=0)

                    if self.loss_type != "gaussian":
                        test_metrics.append(np.zeros_like(test_metrics[-1]))

                    best_test_metrics_repeat.append(np.array(test_metrics[1:]))



                best_metrics_repeat.append(best_metrics)

            best_metrics_repeat = np.stack(best_metrics_repeat)


            mode_weights_path = os.path.join(self.save_weights_path, f"{counter+1}.h5")
            if self.experiment_mode == "normal":
                self.save_weights(cloned_model, mode_weights_path)


            self.Train_Valid_Metric_Results.append(best_metrics_repeat)
            best_metrics_repeat = np.round(best_metrics_repeat.mean(axis=0).mean(axis=-1), 4)

            text = f'''                   Train [ MAE : {best_metrics_repeat[0]}, 
                           RMSE : {best_metrics_repeat[1]}, 
                           R2 : {best_metrics_repeat[2]},
                           STD : {best_metrics_repeat[3]} ], 
                   Valid [ MAE : {best_metrics_repeat[4]}, 
                           RMSE : {best_metrics_repeat[5]},
                           R2 : {best_metrics_repeat[6]}, 
                           STD : {best_metrics_repeat[7]} ]'''
            print(text)


            if self.seperate_test_data:
                best_test_metrics_repeat = np.stack(best_test_metrics_repeat)
                self.Test_Metric_Results.append(best_test_metrics_repeat)

                best_test_metrics_repeat = np.round(best_test_metrics_repeat.mean(axis=0).mean(axis=-1), 4)

                text = f'''                   Test  [ MAE : {best_test_metrics_repeat[0]}, 
                           RMSE : {best_test_metrics_repeat[1]}, 
                           R2 : {best_test_metrics_repeat[2]},
                           STD : {best_test_metrics_repeat[3]} ]'''
                print(text)



        self.Train_Valid_Metric_Results = np.stack(self.Train_Valid_Metric_Results)

        if self.seperate_test_data:
            self.Test_Metric_Results = np.stack(self.Test_Metric_Results)

        if self.experiment_mode == "sensitivity":
            self.Sensitivity_Results = np.stack(self.Sensitivity_Results)