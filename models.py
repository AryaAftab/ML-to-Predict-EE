import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers




# Deep Neural Network
def dnn_model(in_features,
              o_features,
              n_h_layers=4,
              drop_rate=0.25,
              h_units=64,
              activation="relu",
              output_strict=False,
              **kwargs
              ):

    inputs = layers.Input(shape=(in_features,))
    x = inputs

    for _ in range(n_h_layers):
        x = layers.Dense(h_units, activation=activation)(x)
        x = layers.Dropout(drop_rate)(x)

    x = layers.Dense(o_features)(x)

    if output_strict:
        x = layers.Activation("sigmoid")(x)
    outputs = x

    model = keras.models.Model(inputs, outputs)
    
    return model


# Linear Regression
def linear_model(in_features, o_features, output_strict=False, **kwargs):
    inputs = layers.Input(shape=(in_features,))
    x = inputs

    x = layers.Dense(o_features)(x)

    if output_strict:
        x = layers.Activation("sigmoid")(x)
    outputs = x

    model = keras.models.Model(inputs, outputs)
    
    return model


# Polynomial Regression
def polynomial_model(in_features, o_features, degree=10, output_strict=False, **kwargs):
    inputs = layers.Input(shape=(in_features,))

    SumDegree = []
    for counter in range(2, degree+1):
        x = layers.Lambda(lambda x: tf.pow(x, counter))(inputs)
        SumDegree.append(layers.Dense(o_features, use_bias=False)(x))
    SumDegree.append(layers.Dense(o_features, use_bias=True)(inputs))
    
    x = layers.Add()(SumDegree)
    
    if output_strict:
        x = layers.Activation("sigmoid")(x)
    outputs = x

    model = keras.models.Model(inputs, outputs)
    
    return model