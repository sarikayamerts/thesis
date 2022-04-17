from abc import abstractmethod
from unicodedata import name
from tensorflow.keras import models, layers
import tensorflow as tf

class BaseTFModel:
    def __init__(self, window):
        self.window = window
        self.OUT_STEPS = 24

    @property
    def model(self):
        model = models.Sequential()
        model.add(layers.InputLayer(input_shape=(self.window.input_shape)))
        model.add(layers.Permute((1,2,3), name="start------"))

        model = self.add_model(model)

        model.add(layers.Reshape([self.window.number_of_plants, -1], name="end--------"))
        model.add(layers.Dense(self.OUT_STEPS))
        model.add(layers.Permute((2,1)))
        model.add(layers.Reshape([self.OUT_STEPS, self.window.number_of_plants, 1]))
        return model

    @abstractmethod
    def add_model(self, model):
        return model

    def __repr__(self):
        return self.__class__.__name__

