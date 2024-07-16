# Adaptive Learning Rate Methods
import tensorflow as tf
from tensorflow.keras.optimizers import Adagrad, Adadelta, RMSprop, Adam

def train_model_with_adagrad(model, X_train, y_train, learning_rate=0.01):
    optimizer = Adagrad(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mse')
    model.fit(X_train, y_train, epochs=100)
    return model

def train_model_with_adadelta(model, X_train, y_train, learning_rate=1.0):
    optimizer = Adadelta(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mse')
    model.fit(X_train, y_train, epochs=100)
    return model

def train_model_with_rmsprop(model, X_train, y_train, learning_rate=0.001):
    optimizer = RMSprop(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mse')
    model.fit(X_train, y_train, epochs=100)
    return model

def train_model_with_adam(model, X_train, y_train, learning_rate=0.001):
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mse')
    model.fit(X_train, y_train, epochs=100)
    return model
