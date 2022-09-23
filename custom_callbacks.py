import sys
if 'ipykernel' in sys.modules:
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm

import math
import tensorflow as tf
from tensorflow.keras import callbacks





class COLOR:
   PURPLE = '\033[95m'
   CYAN = '\033[96m'
   DARKCYAN = '\033[36m'
   BLUE = '\033[94m'
   GREEN = '\033[92m'
   YELLOW = '\033[93m'
   RED = '\033[91m'
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   END = '\033[0m'



def lr_scheduler(epoch,
                 warmup_epochs=100,
                 decay_epochs=900,
                 initial_lr=1e-6,
                 base_lr=5e-3,
                 min_lr=5e-5):

    if epoch <= warmup_epochs:
        pct = epoch / warmup_epochs
        return ((base_lr - initial_lr) * pct) + initial_lr

    if epoch > warmup_epochs and epoch < warmup_epochs+decay_epochs:
        pct = 1 - ((epoch - warmup_epochs) / decay_epochs)
        return ((base_lr - min_lr) * pct) + min_lr

    return min_lr


class ShowProgress(callbacks.Callback):
    def __init__(self, epochs, step_show=1, metric="loss"):
        super(ShowProgress, self).__init__()
        self.epochs = epochs
        self.step_show = step_show
        self.metric = metric

    def on_train_begin(self, logs=None):
        self.pbar = tqdm(range(self.epochs))

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.step_show == 0:

            self.pbar.set_description(f"""Epoch : {epoch + 1} / {self.epochs}, 
            Train {self.metric} : {round(logs[self.metric], 4)}, 
            Valid {self.metric} : {round(logs['val_' + self.metric], 4)}""")

            self.pbar.update(self.step_show)

            
class BestModelWeights(callbacks.Callback):
    def __init__(self, metric="val_loss", metric_type="min"):
        super(BestModelWeights, self).__init__()
        self.metric = metric
        self.metric_type = metric_type
        if self.metric_type not in ["min", "max"]:
                raise NameError('metric_type must be min or max')

    def on_train_begin(self, logs=None):
        if self.metric_type == "min":
            self.best_metric = math.inf
        else:
            self.best_metric = -math.inf
        self.best_epoch = 0
        self.model_best_weights = None
        
    def on_epoch_end(self, epoch, logs=None):
        if self.metric_type == "min":
            if self.best_metric >= logs[self.metric]:
                self.model_best_weights = self.model.get_weights()
                self.best_metric = logs[self.metric]
                self.best_epoch = epoch
        else:
            if self.best_metric <= logs[self.metric]:
                self.model_best_weights = self.model.get_weights()
                self.best_metric = logs[self.metric]
                self.best_epoch = epoch

    def on_train_end(self, logs=None):
        self.model.set_weights(self.model_best_weights)
        print(COLOR.YELLOW + f"\nBest weights is set, Best Epoch was : {self.best_epoch+1}\n" + COLOR.END)



class Sensitivity(callbacks.Callback):
    def __init__(self,
                 model,
                 main_prediction,
                 valid_dataset,
                 ):
        super(Sensitivity, self).__init__()
        self.model = model
        self.main_prediction = main_prediction
        self.valid_dataset = valid_dataset

        self.best_sensitivity_result = tf.ones(1) * 100

    
    def on_epoch_end(self, epoch, logs=None):

        predicted = self.model.predict(self.valid_dataset[0], verbose=0)

        diff_percent = 100.0 * tf.abs((self.main_prediction - predicted) / (self.main_prediction + 1e-8))
        diff_percent = tf.reduce_mean(diff_percent, axis=0)

        if tf.reduce_mean(diff_percent) <= tf.reduce_mean(self.best_sensitivity_result):
            self.best_sensitivity_result = diff_percent

        logs['sensitivity_result'] = diff_percent

    def on_train_end(self, logs=None):
        print(COLOR.GREEN + f"Best Sensitivity Result : {self.best_sensitivity_result}\n" + COLOR.END)