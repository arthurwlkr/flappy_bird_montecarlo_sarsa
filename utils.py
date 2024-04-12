import numpy as np
import tqdm


def argmax(q_values):
    """
    Argmax function that randomly breaks ties
    """
    top = float("-inf")
    ties = []

    for i in range(len(q_values)):
        if q_values[i] > top:
            top = q_values[i]
            ties = []

        if q_values[i] == top:
            ties.append(i)

    return np.random.choice(ties)


class SarsaAgent:
    """
    State-action-reward-state-action agent
    """
    def __init__(self, env, alpha=0.5, gamma=0.5, epsilon=0.1):
        self.env = env
        self.num_states_x, self.num_states_y = int(20 * 0.7), 24
        self.num_actions = env.action_space.n
        self.Q = np.zeros((self.num_states_x, self.num_states_y, self.num_actions))
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def epsilon_greedy_policy(self, state):
        """
        Usual epsilon-greedy policy
        """
        if np.random.random() < self.epsilon:
            return np.random.randint(self.num_actions)
        else:
            return argmax(self.Q[state])

    def train(self, n_episodes):
        """
        Train the agent for a given number of episodes by updating the Q-values matrix
        """
        train_reward = []
        for episode in tqdm.tqdm(range(n_episodes)):
            state, _ = self.env.reset()
            action = self.epsilon_greedy_policy(state)
            episode_reward = 0

            while True:
                next_state, reward, done, _, _ = self.env.step(action)
                next_action = self.epsilon_greedy_policy(next_state)
                episode_reward += reward

                self.Q[state][action] += self.alpha * (reward + self.gamma * self.Q[next_state][next_action] - self.Q[state][action])

                if done:
                    break

                state = next_state
                action = next_action
            train_reward.append(episode_reward)
        return(train_reward)

    def act(self, state):
        """
        Chose optimal move according to Q-values
        """
        return argmax(self.Q[state])
