#import necessary packages
import gym
import numpy as np
import ray
from really import SampleManager
from gridworlds import GridWorld

"""
Your task is to solve the provided Gridword with tabular Q learning!
In the world there is one place where the agent cannot go, the block.
There is one terminal state where the agent receives a reward.
For each other state the agent gets a reward of 0.
The environment behaves like a gym environment.
Have fun!!!!

"""
#implement the TabularQ algorithm
class TabularQ(object):
    def __init__(self, h, w, action_space):
        self.alpha = 0.95
        self.gamma = 0.99
        self.action_space = action_space
        self.weights = np.full((h,w,4),[0,0,0,0])
        pass

    def __call__(self, state):
        #
        state = state.astype(int)
        state = np.squeeze(state, axis = 0)
        output = {}
        output["q_values"] = np.expand_dims(self.weights[state[0],state[1]],axis = 0)
        return output

    #create get_weights
    def get_weights(self):
        return self.weights
    #create set_weights
    def set_weights(self, q_vals):
        self.weights = q_vals
    #create updateQ
    def updateQ(self, data):
        #extract necessary information from data
        actions = data["action"]
        states = data["state"]
        rewards = data["reward"]
        states_new = data["state_new"]
        not_done = data["not_done"]
        
        #update the q-values accordingly
        #index the Dictionary from the back
        for index in range(len(states)-1,-1,-1):
            curr_state,curr_action = states[index], actions[index]
            q_val_old = self.weights[curr_state]
            q_val_old = q_val_old[actions[index]]
            q_val_new = q_val_old + self.alpha * (rewards[index] + self.gamma*np.max(self.weights[states_new[index]] - q_val_old))
            self.weights[curr_state][curr_action] = q_val_new


if __name__ == "__main__":
    #define env_kwargs
    action_dict = {0: "UP", 1: "RIGHT", 2: "DOWN", 3: "LEFT"}
    env_kwargs = {
        "height": 3,
        "width": 4,
        "action_dict": action_dict,
        "start_position": (2, 0),
        "reward_position": (0, 3),
    }

    #create the environment with defined env_kwargs
    env = GridWorld(**env_kwargs)
    model_kwargs = {"h": env.height, "w": env.width, "action_space": 4}
    
    #define kwargs
    kwargs = {
        "model": TabularQ,
        "environment": GridWorld,
        "num_parallel": 2,
        "total_steps": 100,
        "model_kwargs": model_kwargs
    }
    
    #instantiate manager and the network
    ray.init(log_to_driver=False)
    manager = SampleManager(**kwargs)
    TabQ = TabularQ(env_kwargs["height"],env_kwargs["width"],action_dict)
    
    #train for a number of epochs
    for e in range(3):
        data = manager.get_data(True,100)
        TabQ.updateQ(data)
    
    #apply the trained values to the agent
    manager.set_agent(TabQ.get_weights())
    
    #test the trained network
    manager.test(
        max_steps=200,
        test_episodes=10,
        render=True,
        do_print=True,
        evaluation_measure="time_and_reward",
    )
