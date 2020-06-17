import tensorflow as tf
import game


class Actuator(object):

    def __init__(self):
        self.learning_rate = 1e-4
        self.episodes = 1000

    def train():
        env = game.Game()
        
        for episode in range(self.episodes):
            log_probs = []
            vals = []
            rewards = []

            env.reset()

            with tf.GradientTape() as g:
                for step in range(self.num_steps):
                    state = env.getNpState()
                    state = tf.reshape(state, [g.observation_space[1]])

                    val = critic(state)
                    action_dist = actor(state).numpy()
                    action_dist = np.squeeze(action_dist)

                    possible_actions = env.getPossible()
                    action_dist = [action_dist[i] for i in possible_actions] 
                    action = np.random_choice(possible_actions, p = action_dist)

                    log_prob = np.log(action)
                    entropy = np.sum(np.mean(action_dist) * np.log(action_dist))
                    
                    state, reward = env.step(action)

                    vals.append(val)
                    rewards.append(reward)
                    log_props.append(log_prob)
                    net_entropy += entropy

                    if !env.checkGameActive() or step == self.num_steps - 1:
                        q_val = critic(state)
                        break

                q_vals = np.zeros_like(vals)
                for t in reversed(range(len(rewards))):
                    q_val = rewards[t] + self.gamma * q_val
                    q_vals[t] = q_val

                advantages = q_vals - vals

                actor_loss = -(log_probs * advantages + entropy_term)
                critic_loss = 




    @tf.function
    def step(

