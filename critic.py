import tensorflow as tf
# Import layers:
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout
from tensorflow.keras import Model

#Defining network Below:
class Critic(Model):
  def __init__(self):
    super(Critic, self).__init__()
    # Define layers of the network:
    self.critic_dense_0 = Dense(128, activation='relu')
    self.critic_dense_1 = Dense(64, activation='relu')
    self.critic_dense_2 = Dense(1)

  def call(self, x, training=False):
    # Call layers of network on input x
    # Use the training variable to handle adding layers such as Dropout
    # and Batch Norm only during training
    x = self.critic_dense_0(x)
    x = BatchNormalization()(x)
    if training:
        x = Dropout(.1)(x)
    x = self.critic_dense_1(x)
    x = BatchNormalization()(x)
    if training:
        x = Dropout(.1)(x)
    x = self.critic_dense_2(x)
    return x
