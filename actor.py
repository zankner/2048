import tensorflow as tf
# Import layers:
from tensorflow.keras.layers import Dense
from tensorflow.keras import Model

#Defining network Below:
class Actor(Model):
  def __init__(self, action_dim):
    super(Actor, self).__init__()
    # Define layers of the network:
    self.actor_dense_0 = Dense(128, activation='relu')
    self.actor_dense_1 = Dense(64, activation='relu')
    self.actor_dense_2 = Dense(action_dim, activation='softmax')

  def call(self, x, training=False):
    # Call layers of network on input x
    # Use the training variable to handle adding layers such as Dropout
    # and Batch Norm only during training
    x = self.actor_dense_0(x)
    x = BatchNormalization()(x)
    if training:
        x = Dropout(.1)(x)
    x = self.actor_dense_1(x)
    x = BatchNormalization()(x)
    if training:
        x = Dropout(.1)(x)
    x = self.actor_dense_2(x)
    return x
