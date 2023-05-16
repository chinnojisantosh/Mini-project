import gym
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras import models, layers, optimizers

env = gym.make('CartPole-v0')
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

num_models = 3
ensemble_models = []
for i in range(num_models):
    model = models.Sequential()
    model.add(layers.Dense(24, activation='relu', input_dim=state_size))
    model.add(layers.Dense(24, activation='relu'))
    model.add(layers.Dense(action_size, activation='linear'))
    model.compile(loss='mse', optimizer=optimizers.Adam(learning_rate=0.001))
    ensemble_models.append(model)

max_episodes = 500
max_steps = 200
batch_size = 32
gamma = 0.95
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.995
memory = []

for episode in range(max_episodes):
    state = env.reset()
    state = np.reshape(state, [1, state_size])
    total_reward = 0
    
    for step in range(max_steps):
    # epsilon-greedy strategy
        if np.random.rand() <= epsilon:
            action = random.randrange(action_size)
        else:
            predictions = np.zeros((num_models, action_size))
            for i, model in enumerate(ensemble_models):
                predictions[i] = model.predict(state)[0]
            mean_predictions = np.mean(predictions, axis=0)
            action = np.argmax(mean_predictions)

        
      
        next_state, reward, done, _ = env.step(action)
        next_state = np.reshape(next_state, [1, state_size])
        total_reward += reward
        

        memory.append((state, action, reward, next_state, done))
        state = next_state
        
        # Sampling
        if len(memory) > batch_size:
            batch = random.sample(memory, batch_size)
            
            # Update the weights
            for model in ensemble_models:
                states, actions, rewards, next_states, dones = zip(*batch)
                states = np.concatenate(states)
                next_states = np.concatenate(next_states)
                targets = model.predict(states)
                Q_future = np.amax(model.predict(next_states), axis=1)
                targets[range(batch_size), actions] = rewards + gamma * Q_future * np.invert(dones).astype(np.float32)
                model.fit(states, targets, epochs=1, verbose=0)
                
        if done:
            break
            
  
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay
        

    print("Episode {}: Total Reward = {}".format(episode, total_reward))
