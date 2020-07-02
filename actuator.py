import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import datetime
import numpy as np
import game
import actor
import critic


class Actuator(object):

    def __init__(self):
        self.episodes = 1000
        self.num_steps = 300
        self.gamma = 0.98
        self.actor = actor.Actor(4)
        self.critic = critic.Critic()
        learning_rate = 1e-3
        self.actor_opt = Adam(learning_rate)
        self.critic_opt = Adam(learning_rate)
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        actor_log_dir = 'logs/gradient_tape/' + current_time + '/actor'
        critic_log_dir = 'logs/gradient_tape/' + current_time + '/critic'
        self.actor_summary_writer = tf.summary.create_file_writer(actor_log_dir)
        self.critic_summary_writer = tf.summary.create_file_writer(critic_log_dir)

    def train(self):
        env = game.Game()
        net_entropy = 0
        
        for episode in range(self.episodes):
            log_probs = []
            vals = []
            rewards = []

            env.reset()

            with tf.GradientTape(persistent=True) as tape:
                for step in range(self.num_steps):
                    state = env.getNpState()
                    state = tf.reshape(state, [1, env.observation_space[1]])
                    val = self.critic(state, training=True)
                    action_dist = self.actor(state, training=True)
                    action_dist_np = action_dist.numpy()
                    action_dist_np = np.squeeze(action_dist_np)

                    possible_actions = env.getPossible()
                    action_dist_np = [action_dist_np[i] for i in possible_actions]
                    action_dist_np /= np.sum(action_dist_np)
                    action = np.random.choice(possible_actions, p = action_dist_np)
                    test = tf.gather(tf.squeeze(action_dist), [action])
                    log_prob = tf.math.log(test)
                    entropy = tf.math.reduce_sum(
                        tf.math.reduce_mean(action_dist) * tf.math.log(action_dist))
                    
                    state, reward, active = env.step(action)

                    vals.append(val)
                    rewards.append(reward)
                    log_probs.append(log_prob)
                    net_entropy += entropy

                    if not active or step == self.num_steps - 1:
                        state = tf.reshape(state, [1,16])
                        q_val = self.critic(state, training=True)
                        break

                q_vals = [0 for i in range(len(rewards))]
                for t in reversed(range(len(rewards))):
                    q_val = rewards[t] + self.gamma * q_val
                    q_vals[t] = q_val

                vals = tf.convert_to_tensor(vals)
                rewards = tf.convert_to_tensor(rewards)
                log_probs = tf.convert_to_tensor(log_probs)
                q_vals = tf.convert_to_tensor(q_vals)

                advantages = q_vals - vals

                actor_loss = tf.math.reduce_mean(-log_probs * advantages) - 1e-4 * net_entropy
                critic_loss = tf.math.reduce_mean(tf.math.pow(advantages, 2)) - 1e-4 * net_entropy
            
            gradients = tape.gradient(actor_loss, self.actor.trainable_variables)
            self.actor_opt.apply_gradients(zip(gradients, self.actor.trainable_variables))

            gradients = tape.gradient(critic_loss, self.critic.trainable_variables)
            self.critic_opt.apply_gradients(zip(gradients, self.critic.trainable_variables))

            del tape

            self._log(actor_loss, critic_loss, episode)

            if episode % 10 == 0:
                print(f'Avg reward: {np.mean(rewards)}')


    @tf.function
    def _update(self, loss, tape, optimizer, model):
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    
    def _log(self, actor_loss, critic_loss, epoch):
        with self.actor_summary_writer.as_default():
            tf.summary.scalar('actor', actor_loss, step=epoch)
        with self.critic_summary_writer.as_default():
            tf.summary.scalar('critic', critic_loss, step=epoch)


a = Actuator()
a.train()
