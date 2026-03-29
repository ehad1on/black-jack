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

Q = np.zeros(5)

learning_rate = 0.1
discount_factor = 0.9
episodes = 100

for episode in range(episodes):
    position = 3  # Start in the middle position (state)
    #time = 0
    while True:
        action = random.randint(0, 1)  # Pick random action
        #time += 1
        # Simulate environment response
        if action == 0 and position > 0:  # Move left
            position = max(0, position - 1)
        elif action == 1 and position < 4:  # Move right
            position = min(4, position + 1)

        reward = 2 if position == 4 else -1  # Reward for moving right to the edge

        # Q-learning update formula:
        # Q(state, action) = Q(state, action) + learning_rate * (reward + discount_factor * max_future_Q - current_Q)
        old_value = Q[position]
        max_future = np.max([Q[position - 1] if position > 0 else 0, Q[position + 1] if position < 4 else 0])  # Best possible next move
        new_value = old_value + learning_rate * (reward + discount_factor * max_future - old_value)
        Q[position] = new_value

        if position == 4:
            break  # Episode ends if we move right to the edge

    print(f"Episode {episode + 1}: q-values = {Q}")

print("Learned Q-values:", Q)
print("Best action:", np.argmax(Q))  # Action 2 (right) should have highest value

