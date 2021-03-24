#import necessary packages
import gym
import numpy as np
import ray
from really import SampleManager
import tensorflow as tf
import tensorflow_probability as tfp

#create network for the policy
class Policy_Network(tf.keras.Model):
    def __init__(self):
        super(Policy_Network, self).__init__()
        #create layers for learning with 4 unit output layer for 2 mus and 2 sigmas
        self.p_layers =  [tf.keras.layers.Dense(units = 64, activation = "sigmoid"),
                        tf.keras.layers.Dense(units = 32, activation = "sigmoid"),
                        tf.keras.layers.Dense(units = 4, use_bias = False)]

    def __call__(self, inp):
        for layer in self.p_layers:
            inp = layer(inp)
        #define first 2 output values as the 2 mus
        mu = inp[0][:2]
        #define other 2 values as the sigmas
        sigma = tf.math.sqrt(tf.math.square(inp[0][2:]))
        return mu, sigma

#create network to estimate the advantages
class Value_Network(tf.keras.Model):
    def __init__(self):
        super(Value_Network, self).__init__()
        #create layers for learning with 2 output units for 2 estimates for both distributions of policy_network
        self.v_layers =  [tf.keras.layers.Dense(units = 64, activation = "sigmoid"),
                        tf.keras.layers.Dense(units = 32, activation = "sigmoid"),
                        tf.keras.layers.Dense(units = 2, use_bias = False)]

    def __call__(self, inp):
        for layer in self.v_layers:
            inp = layer(inp)
        return inp

#combine both networks as PPO
class PPO(tf.keras.Model):
    def __init__(self):
        super(PPO, self).__init__()

        #use the value network
        self.value = Value_Network()

        #use the policy network for the old and current policy for comparison later
        self.policy = Policy_Network()
        self.old_policy = Policy_Network()

    def __call__(self,inp):
        #create the output in the correct form
        output = {}
        output["mu"], output["sigma"] = self.policy(inp)
        output["value_estimate"] = self.value(inp)
        return output

    #We implemented the set_weights method to give pretrained weights to the manager for testing
    #We're not sure why but the get_weights method returned a list of length 15 but the pre-implemented set_weights method needed only a length of 10.
#     def set_weights(self, weights):
#         self.value.set_weights(weights[:5])
#         self.policy.set_weights(weights[5:10])
        #self.old_policy.set_weights(weights[10:])

#implement the hyperparameters for training
gamma = 0.95
epsilon = 1e-6
KL = 0.9

#use the training method with different optimizers and learning rates for the 2 networks
def training(agent, manager, epochs = 1, learning_rate_p = 0.001, learning_rate_v = 0.01):
    p_optimizer = tf.keras.optimizers.Adam(learning_rate_p)
    v_optimizer = tf.keras.optimizers.Adam(learning_rate_v)
    #create a buffer to sample from
    manager.initilize_buffer(1000)
    for epoch in range(epochs):
        data = manager.get_data(True,1000)
        manager.store_in_buffer(data)
        #sample 64 samples from buffer repeatedly
        for i in range(100):
            sample = manager.sample(64, from_buffer = True)
            #extract necessary values from the samples
            state = sample["state"]
            state_new = sample["state_new"]
            action = sample["action"]
            #reshape the rewards because 2 distributions and advatages are estimated each step
            rewards = np.stack((sample["reward"],sample["reward"]), axis = -1)

            #create and compare the old and current policy to check for our constraint
            mu, sigma = agent.model.policy(state)
            pi = tfp.distributions.Normal(mu, sigma)

            old_mu, old_sigma = agent.model.old_policy(state)
            old_pi = tfp.distributions.Normal(old_mu, old_sigma)

            scale = pi.prob(action) / old_pi.prob(action)

            #if the constraint is satisfied update the agent
            #if np.asarray(scale).any() >= KL:
            with tf.GradientTape() as tape:
                #compute loss for value_network
                reward_v = tf.convert_to_tensor(rewards, dtype=np.float32)
                advantage_v = reward_v - agent.model.value(state)
                loss_v = tf.reduce_mean(tf.square(advantage_v))
                #compute gradients for value_network
                gradients_v = tape.gradient(loss_v, agent.model.value.trainable_variables)

            with tf.GradientTape() as tape:
                #compute loss for policy_network
                mu, sigma = agent.model.policy(state)
                pi = tfp.distributions.Normal(mu, sigma)

                old_mu, old_sigma = agent.model.old_policy(state)
                old_pi = tfp.distributions.Normal(old_mu, old_sigma)

                scale = pi.prob(action) / old_pi.prob(action)
                advantage_p = rewards + agent.v(state_new) - agent.v(state)
                s = scale * advantage_p

                loss_p = -tf.reduce_mean(tf.minimum(s, tf.clip_by_value(scale,1. - epsilon, 1. + epsilon)* advantage_p))
                #compute gradients for policy_network from old policy
                gradients_p = tape.gradient(loss_p, agent.model.old_policy.trainable_variables)

            #apply the computed gradients to the networks separately
            v_optimizer.apply_gradients(zip(gradients_v, agent.model.value.trainable_variables))
            p_optimizer.apply_gradients(zip(gradients_p, agent.model.policy.trainable_variables))
        agent.model.old_policy.set_weights(agent.model.policy.get_weights())
        print("epoch: "+ str(epoch+1))
    # print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    # print(agent.get_weights())


if __name__ == "__main__":
    #create environment
    env = gym.make("LunarLanderContinuous-v2")
    model_kwargs = {}
    #define kwargs
    kwargs = {
        "model": PPO,
        "environment": "LunarLanderContinuous-v2",
        "num_parallel": 2,
        "total_steps": 100,
        "model_kwargs": model_kwargs,
        #use continuous normal diagonal as sampling type because of the continuous action space
        "action_sampling_type": "continuous_normal_diagonal",
        #use a value_estimate
        "value_estimate": True
    }

    #instantiate the manager and agent for training
    ray.init(log_to_driver=False)
    manager = SampleManager(**kwargs)
    env.reset()

    #train for predefined number of epochs
    agent = manager.get_agent()
    training(agent,manager,epochs = 30)

    #unfortunately we were not able to make the manager's test method work because we were unable to pass over the trained agent's weights

    # print("*********************************************************************")
    # print("weights of during testing : {}".format(agent.model.get_weights()))
    #
    #
    # manager.set_agent(agent.model.get_weights())
    # print("testing")
    # manager.test(
    #     max_steps=1000,
    #     test_episodes=50,
    #     render=True,
    #     do_print=True,
    #     evaluation_measure="time_and_reward",
    #  )
