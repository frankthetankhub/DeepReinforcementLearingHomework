#import necessary packages
import gym
import numpy as np
import ray
from really import SampleManager
#from gridworlds import GridWorld
import tensorflow as tf

#create the DQN network
class DQN(tf.keras.Model):
    def __init__(self):
        super(DQN,self).__init__()
        #use multiple dense layers to estimate the q_values
        self.Layers = [tf.keras.layers.Dense(units = 64, activation = "sigmoid"),
                      tf.keras.layers.Dense(units = 32, activation = "sigmoid"),
                       tf.keras.layers.Dense(units = 16, activation = "sigmoid"),
                        tf.keras.layers.Dense(units = 2, use_bias = False)]
        
    def __call__(self,inp):
        for layer in self.Layers:
            inp = layer(inp)
        output = {}
        output["q_values"] = inp
        return output

#set hyperparameter
gamma = 0.95

#use the training method to train the network
def training(agent, manager, epochs = 1, learning_rate = 0.01):
    manager.initilize_buffer(1000)
    #create a buffer to sample from
    optimizer = tf.keras.optimizers.Adam(learning_rate)
    for epoch in range(epochs):
        #get Trainingdata once per Epoch and store in Buffer
        data = manager.get_data(True, 1000)
        manager.store_in_buffer(data)
        #sample 64 samples from the buffer repeatedly
        for i in range(100):            
            with tf.GradientTape() as tape:
                sample = manager.sample(64)
                #extract the necessary values from the samples
                state = sample["state"]
                state_new = sample["state_new"]
                action = sample["action"]
                rewards = sample["reward"]
                
                #compute the loss
                loss =  tf.math.square(rewards + gamma * agent.max_q(state_new) - agent.q_val(state,action))
                
                #compute gradients
                gradients = tape.gradient(loss, agent.model.trainable_variables)
            #apply gradients
            optimizer.apply_gradients(zip(gradients, agent.model.trainable_variables))
        print("epoch: "+ str(epoch+1))

if __name__ == "__main__":
    #create environment
    env = gym.make("CartPole-v0")
    model_kwargs = {}
    #define kwargs
    kwargs = {
        "model": DQN,
        "environment": "CartPole-v0",
        "num_parallel": 2,
        "total_steps": 100,
        "model_kwargs": model_kwargs,
    }

    #instantiate the manager and agent for training
    ray.init(log_to_driver=False)
    manager = SampleManager(**kwargs)
    env.reset()
    agent = manager.get_agent()
    
    #train for the number of epochs
    training(agent,manager,epochs = 30)

    #test the trained network
    manager.set_agent(agent.model.get_weights())
    manager.test(
        max_steps=10000,
        test_episodes=100,
        render=True,
        do_print=True,
        evaluation_measure="time_and_reward",
    )
