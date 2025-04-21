import numpy as np
import matplotlib.pyplot as plt

def plot_policy_and_value(pi, V, iteration):
    V_grid = V.reshape((10, 10))
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(V_grid, cmap='coolwarm', extent=[-0.5, 9.5, 9.5, -0.5])

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Value")

    delta_dict = {
        0: (1, 0),  
        1: (0, 1),   
        2: (0, -1),  
        3: (-1, 0)    
    }

    for s in range(100):
        r, c = divmod(s, 10)
        if (r, c) in OBSTACLES:
            rect = plt.Rectangle((c - 0.5, r - 0.5), 1, 1, color='gray')
            ax.add_patch(rect)
            continue
        if (r, c) == goal_coord:
            rect = plt.Rectangle((c - 0.5, r - 0.5), 1, 1, color='green')
            ax.add_patch(rect)
            continue
        if (r, c) == start_coord:
            rect = plt.Rectangle((c - 0.5, r - 0.5), 1, 1, color='blue')
            ax.add_patch(rect)
        
        action = pi[s]
        dx, dy = delta_dict[action][1] * 0.3, -delta_dict[action][0] * 0.3
        ax.arrow(c, r, dx, dy, head_width=0.2, head_length=0.2, fc='black')

        ax.text(c, r, f"{V[s]:.1f}", va='center', ha='center', color='white', fontsize=8)

    ax.set_xticks(np.arange(0, 10, 1))
    ax.set_yticks(np.arange(0, 10, 1))
    ax.set_title(f"Iteration {iteration}")
    ax.set_xlim([-0.5, 9.5])
    ax.set_ylim([9.5, -0.5])
    plt.grid(True)
    plt.show()


pi = 1 * np.ones(100, dtype=int)  

start_coord = (3, 6)   
goal_coord = (8, 1)    

OBSTACLES = [(3, 2), (4, 2), (5, 2), (6, 2),
                      (4, 4), (4, 5), (4, 6), (4, 7),
                      (5, 7), (7, 5), (7, 4)]


OBSTACLES = set(OBSTACLES)
for i in range(10):
    OBSTACLES.add((i, 0))            
    OBSTACLES.add((i, 9)) 
    OBSTACLES.add((0, i))          
    OBSTACLES.add((9, i)) 

actions = ['N', 'E', 'W', 'S']
action_to_delta = {
    'N': (-1, 0),
    'E': (0, 1),
    'W': (0, -1),
    'S': (1, 0)
}

outcome_probs = {
    'N': {(-1, 0): 0.7, (0, -1): 0.1, (0, 1): 0.1, (0, 0): 0.1},
    'E': {(0, 1): 0.7, (-1, 0): 0.1, (1, 0): 0.1, (0, 0): 0.1},
    'W': {(0, -1): 0.7, (-1, 0): 0.1, (1, 0): 0.1, (0, 0): 0.1},
    'S': {(1, 0): 0.7, (0, 1): 0.1, (0, -1): 0.1, (0, 0): 0.1}
}
def get_next_state(state_coord, delta):
    r, c = state_coord
    dr, dc = delta
    new_r, new_c = r + dr, c + dc

    # Check grid boundaries first.
    if new_r < 0 or new_r >= 10 or new_c < 0 or new_c >= 10:
        return state_coord  
    
    # Check for obstacles.
    if (new_r, new_c) in OBSTACLES:
        return state_coord

    return (new_r, new_c)


T = np.zeros((len(actions), 100, 100))
for a_idx, action in enumerate(actions):
    for r in range(10):
        for c in range(10):
            current_coord = (r, c)
            s = r * 10 + c
            if current_coord in OBSTACLES:
                T[a_idx, s, s] = 1.0
                continue

            for delta, prob in outcome_probs[action].items():
                next_coord = get_next_state(current_coord, delta)
                s_next = next_coord[0] * 10 + next_coord[1]
                T[a_idx, s, s_next] += prob



def reward(s):
    r, c = divmod(s, 10)
    if (r, c) == goal_coord:
        return 10
    elif (r, c) in OBSTACLES:
        return -10
    else:
        return -1
    

def policy_evaluation(pi, V, T, gamma=0.9, theta=1e-3):
    while True:
        delta = 0
        for s in range(100):
            v = V[s]
            V[s] = sum(T[pi[s], s, s_next] * (reward(s_next) + gamma * V[s_next]) for s_next in range(100))
            delta = max(delta, abs(v - V[s]))
        if delta < theta:
            break
    return V

def policy_improvement(pi, V, T, gamma=0.9):
    policy_stable = True
    for s in range(100):
        old_action = pi[s]
        action_values = np.zeros(len(actions))
        for a in range(len(actions)):
            action_values[a] = sum(T[a, s, s_next] * (reward(s_next) + gamma * V[s_next]) for s_next in range(100))
        pi[s] = np.argmax(action_values)
        if old_action != pi[s]:
            policy_stable = False
    return policy_stable

def policy_iteration():
    pi = np.ones(100, dtype=int)
    V = np.zeros(100)
    iteration = 0
    while True:
        print(f"Policy Iteration Step {iteration}")
        plot_policy_and_value(pi, V, iteration)
        V = policy_evaluation(pi, V, T)
        if policy_improvement(pi, V, T):
            plot_policy_and_value(pi, V, iteration + 1)  
            break
        iteration += 1
    return pi, V


final_pi, final_V = policy_iteration()

print("Final Policy:")
print(final_pi.reshape((10, 10)))
print("Final Value Function:")
print(final_V.reshape((10, 10)))





