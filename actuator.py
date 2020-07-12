import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import categorical_crossentropy, Huber
import datetime
import numpy as np
import game
import actor
import critic
import math


class Actuator(object):

    def __init__(self):
        self.episodes = 1000
        self.gamma = 0.99
        self.actor = actor.Actor(4)
        self.critic = critic.Critic()
        actor_learning_rate = 1e-4
        critic_learning_rate = 1e-3
        self.actor_opt = Adam(actor_learning_rate)
        self.critic_opt = Adam(critic_learning_rate)
        self.critic_loss = Huber()
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        actor_log_dir = 'logs/gradient_tape/' + current_time + '/actor'
        critic_log_dir = 'logs/gradient_tape/' + current_time + '/critic'
        reward_log_dir = 'logs/gradient_tape/' + current_time + '/reward'
        self.actor_summary_writer = tf.summary.create_file_writer(actor_log_dir)
        self.critic_summary_writer = tf.summary.create_file_writer(critic_log_dir)
        self.reward_summary_writer = tf.summary.create_file_writer(reward_log_dir)
        self.actor_save_dir = 'saved_models/' + current_time + '/actor'
        self.critic_save_dir = 'saved_models/' + current_time + '/critic'

    def train(self):
        env = game.Game()

        for episode in range(100000):
            active = True
            entropy = tf.Variable(0.0)

            env.reset()

            rewards = []
            log_probs = []
            vals = []

            counter = 0
            with tf.GradientTape(persistent=True) as tape:
                while active:
                    state = env.getNpState()
                    state = tf.expand_dims(state, 0)
                    action_logits = self.actor(state, training=True)
                    state_val = self.critic(state, training=True)

                    possible_actions = env.getPossible()
                    action_mask = [[action in possible_actions for action in range(4)]]
                    masked_action_logits = tf.expand_dims(tf.boolean_mask(action_logits, action_mask), 0)
                    masked_action_dist = tf.nn.softmax(masked_action_logits)
                    possible_index = tf.squeeze(tf.random.categorical(
                        tf.math.log(masked_action_dist), 1)).numpy()
                    prob = tf.squeeze(tf.gather(masked_action_dist, [possible_index], axis=1))
                    log_prob = tf.math.log(prob)
                    
                    entropy = entropy + categorical_crossentropy(
                            masked_action_dist, masked_action_dist)

                    action = possible_actions[possible_index]

                    state, reward, active = env.step(action)

                    log_probs.append(log_prob)
                    rewards.append(reward)
                    vals.append(state_val)

                print('*' * 6)
                for row in env.board:
                    print(row)

                rewards[-1] -= 10


                state = tf.expand_dims(state, 0)
                q_val = self.critic(state, training=True)

                q_vals = [0 for i in range(len(rewards))]
                for t in reversed(range(len(rewards))):
                    q_val = rewards[t] + self.gamma * q_val
                    q_vals[t] = q_val

                log_probs = tf.convert_to_tensor(log_probs)
                vals = tf.convert_to_tensor(vals)
                q_vals = tf.convert_to_tensor(q_vals)

                q_vals = (q_vals - np.mean(q_vals)) / (np.std(q_vals) + 1e-6)

                advantages = q_vals - vals


                log_probs = tf.convert_to_tensor(log_probs)
                advantages = tf.convert_to_tensor(advantages)

                actor_loss = -tf.reduce_mean(log_probs * advantages) - 1e-4 * entropy
                actor_loss = tf.squeeze(actor_loss)
                critic_loss = tf.reduce_mean(self.critic_loss(vals, q_vals))

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

    def _save(self):
        self.actor.save(self.actor_save_dir, include_optimizer=False, save_format='tf')
        self.critic.save(self.critic_save_dir, include_optimizer=False, save_format='tf')


a = Actuator()
a.train()
a._save()
