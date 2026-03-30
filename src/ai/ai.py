# code von copilot (simplest q-learning example)
'''
import numpy as np
import random

# 3x3 grid, 4 possible actions (up, down, left, right)
grid_size = 3
num_actions = 4
num_states = grid_size * grid_size

Q = np.zeros((num_states, num_actions))  # Q-table
learning_rate = 0.1
discount_factor = 0.95
episodes = 1000

for episode in range(episodes):
    state = random.randint(0, num_states - 1)  # Random starting position
    
    while True:
        # Pick random action
        action = random.randint(0, num_actions - 1)
        
        # Calculate next state
        row, col = state // grid_size, state % grid_size
        if action == 0: row = max(0, row - 1)      # up
        elif action == 1: row = min(grid_size-1, row + 1)  # down
        elif action == 2: col = max(0, col - 1)    # left
        elif action == 3: col = min(grid_size-1, col + 1)  # right
        
        next_state = row * grid_size + col
        
        # Reward: +1 at goal (bottom-right), -0.01 per step
        reward = 1 if next_state == num_states - 1 else -0.01
        
        # Q-learning update
        old_value = Q[state, action]
        max_future = np.max(Q[next_state])
        Q[state, action] = old_value + learning_rate * (reward + discount_factor * max_future - old_value)
        
        state = next_state
        if state == num_states - 1:  # Reached goal
            break

print("Agent learned!")
'''
# experimenting (modifing copilot code)
import numpy as np
import random

Q = np.zeros((5,5))

learning_rate = 0.1
discount_factor = 0.9
episodes = 2000

for episode in range(episodes):
    position = (2,2)  # Start in the middle position (state)
    #time = 0
    while True:
        if episode < 500:  # Print every 10 episodes
            action = random.randint(0, 3)  # Pick random action
        else :  # After 50 episodes, choose the best action based on Q-values
            action = np.argmax([Q[(position[0], position[1] - 1)] if position[1] > 0 else -np.inf, Q[(position[0], position[1] + 1)] if position[1] < 4 else -np.inf, Q[(position[0] - 1, position[1])] if position[0] > 0 else -np.inf, Q[(position[0] + 1, position[1])] if position[0] < 4 else -np.inf])  # Best action based on Q-values
            #print(action)
        #time += 1
        # Simulate environment response
        if action == 0:  # Move left
            position = (position[0], max(0, position[1] - 1))
        elif action == 1:  # Move right
            position = (position[0], min(4, position[1] + 1))
        elif action == 2:  # Move up
            position = (max(0, position[0] - 1), position[1])
        elif action == 3:  # Move down
            position = (min(4, position[0] + 1), position[1])

        if position == (4,4):
            reward = 1
        elif position in [(3,1), (3,2), (3,3), (3,4)]:
            reward = -9
        else:
            reward = -1

        # Q-learning update formula:
        # Q(state, action) = Q(state, action) + learning_rate * (reward + discount_factor * max_future_Q - current_Q)
        old_value = Q[position]
        max_future = np.max([Q[(position[0], position[1] - 1)] if position[1] > 0 else -np.inf, Q[(position[0], position[1] + 1)] if position[1] < 4 else -np.inf, Q[(position[0] - 1, position[1])] if position[0] > 0 else -np.inf, Q[(position[0] + 1, position[1])] if position[0] < 4 else -np.inf])  # Best possible next move
        new_value = old_value + learning_rate * (reward + discount_factor * max_future - old_value)
        Q[position] = new_value

        if position == (4,4):
            break  # Episode ends if we move right to the edge

    #print(f"Episode {episode + 1}: q-values = \n{Q}")

print("Learned Q-values:\n", Q)
#can i color the values?
try:
    import matplotlib.pyplot as plt
    plt.imshow(Q, cmap='viridis', interpolation='nearest')
    plt.colorbar()
    plt.title("Learned Q-values")
    plt.show()
except ImportError:
    print("Matplotlib not installed, cannot visualize Q-values.")

