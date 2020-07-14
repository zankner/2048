import tensorflow as tf
# Import layers:
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras import Model

#Defining network Below:
class Critic(Model):
  def __init__(self):
    super(Critic, self).__init__()
    # Define layers of the network:
    self.critic_conv_0 = Conv2D(256,1, activation='relu')
    self.critic_conv_1 = Conv2D(128,1, activation='relu')
    self.critic_pool_0 = MaxPooling2D(2)
    self.critic_pool_1 = MaxPooling2D(2)
    self.critic_flatten_0 = Flatten()
    self.critic_dense_0 = Dense(256, activation='relu')
    self.critic_dense_1 = Dense(256, activation='relu')
    self.critic_dense_2 = Dense(128, activation='relu')
    self.critic_dense_3 = Dense(64, activation='relu')
    self.critic_dense_4 = Dense(1)
    self.norm_0 = BatchNormalization()
    self.norm_1 = BatchNormalization()
    self.norm_2 = BatchNormalization()
    self.norm_3 = BatchNormalization()

  def call(self, x, training=False):
    x = self.critic_dense_0(x)
    if training:
        x = Dropout(.2)(x)
    x = self.norm_0(x)
    x = self.critic_dense_1(x)
    if training:
        x = Dropout(.2)(x)
    x = self.norm_1(x)
    x = self.critic_dense_2(x)
    if training:
        x = Dropout(.2)(x)
    x = self.norm_2(x)
    x = self.critic_dense_4(x)
    return x
