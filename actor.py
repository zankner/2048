import tensorflow as tf
# Import layers:
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras import Model

#Defining network Below:
class Actor(Model):
  def __init__(self, action_dim):
    super(Actor, self).__init__()
    # Define layers of the network:
    self.actor_conv_0 = Conv1D(16,2, activation='relu')
    self.actor_pool_0 = MaxPooling1D(2, 1)
    self.actor_flatten_0 = Flatten()
    self.actor_dense_0 = Dense(512, activation='relu')
    self.actor_dense_1 = Dense(256, activation='relu')
    self.actor_dense_2 = Dense(32, activation='relu')
    self.actor_dense_3 = Dense(16, activation='relu')
    self.actor_dense_4 = Dense(action_dim)
    self.norm_0 = BatchNormalization()
    self.norm_1 = BatchNormalization()
    self.norm_2 = BatchNormalization()
    self.norm_3 = BatchNormalization()

  def call(self, x, training=False):
    # Call layers of network on input x
    # Use the training variable to handle adding layers such as Dropout
    # and Batch Norm only during training
    #x = self.norm_0(x)
    #x = self.actor_dense_0(x)
    #if training:
    #    x = Dropout(.2)(x)
    #x = self.norm_1(x)
    #x = self.actor_dense_1(x)
    #if training:
    #    x = Dropout(.2)(x)
    #x = self.norm_2(x)
    #x = self.actor_dense_2(x)
    #if training:
    #    x = Dropout(.2)(x)
    #x = self.norm_3(x)
    #x = self.actor_dense_3(x)
    #if training:
    #    x = Dropout(.3)(x)
    #x = self.norm_0(x)
    #x = self.actor_dense_4(x)
    x = self.actor_conv_0(x)
    x = self.actor_pool_0(x)
    x = self.actor_flatten_0(x)
    if training:
        x = Dropout(.3)(x)
    x = self.norm_0(x)
    x = self.actor_dense_4(x)
    return x
