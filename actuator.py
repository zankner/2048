import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import datetime
import numpy as np
import game
import actor
import critic
import gym


class Actuator(object):

    def __init__(self):
        self.episodes = 1000
        self.gamma = 0.99
        self.actor = actor.Actor(2)
        self.critic = critic.Critic()
        actor_learning_rate = 1e-4
        critic_learning_rate = 1e-2
        self.actor_opt = Adam(actor_learning_rate)
        self.critic_opt = Adam(critic_learning_rate)
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        actor_log_dir = 'logs/gradient_tape/' + current_time + '/actor'
        critic_log_dir = 'logs/gradient_tape/' + current_time + '/critic'
        reward_log_dir = 'logs/gradient_tape/' + current_time + '/reward'
        self.actor_summary_writer = tf.summary.create_file_writer(actor_log_dir)
        self.critic_summary_writer = tf.summary.create_file_writer(critic_log_dir)
        self.reward_summary_writer = tf.summary.create_file_writer(reward_log_dir)

    def train(self):
        env = gym.make("CartPole-v0")

        for episode in range(1):
            with tf.GradientTape(persistent=True) as tape:
                rewards = []
                log_probs = []
                vals = []
                done = False

                state = env.reset()

                for i in range(3):
                    state = tf.expand_dims(state, 0)
                    action_dist = self.actor(state)
                    state_val = self.critic(state)

                    action = tf.squeeze(tf.random.categorical(
                        tf.math.log(action_dist), 1)).numpy()
                    prob = tf.squeeze(tf.gather(action_dist, [action], axis=1))
                    log_prob = tf.math.log(prob)

                    state, reward, done, _ = env.step(action)

                    log_probs.append(log_prob)
                    rewards.append(reward)
                    vals.append(state_val)

                state = tf.expand_dims(state, 0)
                q_val_terminal = self.critic(state)

                q_vals = []
                for i in range(len(rewards) - 1):
                    q_vals.append(rewards[i] + self.gamma * vals[i + 1])
                q_vals.append(rewards[-1] + self.gamma * q_val_terminal)

                log_probs = tf.convert_to_tensor(log_probs)
                vals = tf.convert_to_tensor(vals)
                q_vals = tf.convert_to_tensor(q_vals)

                advantages = q_vals - vals

                log_probs = tf.convert_to_tensor(log_probs)
                advantages = tf.convert_to_tensor(advantages)

                actor_loss = -tf.reduce_mean(log_probs * advantages)
                critic_loss = tf.reduce_mean(tf.math.pow(advantages, 2))

            actor_gradients = tape.gradient(actor_loss, self.actor.trainable_variables)
            self.actor_opt.apply_gradients(zip(actor_gradients, self.actor.trainable_variables))

            critic_gradients = tape.gradient(critic_loss, self.critic.trainable_variables)
            self.critic_opt.apply_gradients(zip(critic_gradients, self.critic.trainable_variables))

            del tape

            self._log(actor_loss, critic_loss, np.sum(rewards), episode)


    @tf.function
    def _update(self, loss, tape, optimizer, model):
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    
    def _log(self, actor_loss, critic_loss, reward, epoch):
        with self.actor_summary_writer.as_default():
            tf.summary.scalar('actor', actor_loss, step=epoch)
        with self.critic_summary_writer.as_default():
            tf.summary.scalar('critic', critic_loss, step=epoch)
        with self.reward_summary_writer.as_default():
            tf.summary.scalar('reward', reward, step=epoch)


a = Actuator()
a.train()
