import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import categorical_crossentropy
import datetime
import numpy as np
import game
import actor
import critic


class Actuator(object):

    def __init__(self):
        self.episodes = 1000
        self.gamma = 0.98
        self.actor = actor.Actor(4)
        self.critic = critic.Critic()
        actor_learning_rate = 1e-3
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
        self.actor_save_dir = 'saved_models/' + current_time + '/actor'
        self.critic_save_dir = 'saved_models/' + current_time + '/critic'

    def train(self):
        env = game.Game()

        for episode in range(1000):
            active = True
            entropy = tf.Variable(0.0)

            net_reward = 0
            net_critic_loss = 0
            net_actor_loss = 0

            env.reset()

            counter = 0
            while active:
                with tf.GradientTape(persistent=True) as tape:
                    rewards = []
                    log_probs = []
                    vals = []
                    for step in range(200):
                        counter +=1
                        state = env.getNpState()
                        state = tf.expand_dims(state, 0)
                        action_logits = self.actor(state)
                        state_val = self.critic(state)

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

                        if not active:
                            print('*' * 6)
                            for row in env.board:
                                print(row)
                            break

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

                    actor_loss = -tf.reduce_mean(log_probs * advantages) - 1e-4 * entropy
                    actor_loss = tf.squeeze(actor_loss)
                    critic_loss = tf.reduce_mean(tf.math.pow(advantages, 2))

                    net_reward += np.sum(rewards)
                    net_actor_loss += actor_loss.numpy()
                    net_critic_loss += critic_loss.numpy()

                actor_gradients = tape.gradient(actor_loss, self.actor.trainable_variables)
                self.actor_opt.apply_gradients(zip(actor_gradients, self.actor.trainable_variables))

                critic_gradients = tape.gradient(critic_loss, self.critic.trainable_variables)
                self.critic_opt.apply_gradients(zip(critic_gradients, self.critic.trainable_variables))

                del tape

            self._log(net_actor_loss, net_critic_loss, net_reward, episode)


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
