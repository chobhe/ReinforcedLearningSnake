import random
import numpy as np
import pickle
import RLSnake
import matplotlib
import matplotlib.pyplot as plt


import tensorflow as tf

from tf_agents.agents.dqn import dqn_agent
from tf_agents.drivers import dynamic_step_driver
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.networks import q_network
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.utils import common

class Computer:
    def __init__(self):
        #hyperparameters for tuning
        #controls the learning process, other parameters derived through training
        #number of runs
        self.num_iterations = 14000
        #initial amount of data collections, seems to boost learning a lot
        self.initial_collect_steps = 1000
        #every run we collect this number of times
        self.collect_steps_per_iteration = 1

        self.replay_buffer_max_length = 100000

        self.batch_size = 64
        self.learning_rate = 1e-3
        self.log_interval = 500

        self.num_eval_episodes = 5
        self.eval_interval = 2000

    def train_eval(self):
        training_env = RLSnake.RLSnake()
        #put the environment into pickle
        evaluating_env = RLSnake.RLSnake(pickle_eval = True)
        #wraps numpy arrays to tensors to work with tensorflow agents
        train_env = tf_py_environment.TFPyEnvironment(training_env)
        eval_env = tf_py_environment.TFPyEnvironment(evaluating_env)

        fc_layer_params = (100,)

        q_net = q_network.QNetwork(
        #creates a q network with the observation spec the action spec and the size of the models hidden layers
            train_env.observation_spec(),
            train_env.action_spec(),
            fc_layer_params=fc_layer_params)


        #creates a deep q network agent
        #requires a time_step_spec, action_spec, q network, optimizer, a loss function and a step counter
        #adam is an algorithm that optimizes the learning rate so we don't get caught on local minima, it's an adaptive learning rate that changes
        #the learning rate during training
        #SGD which is standard gradient descent is a nonadaptave rate which we toggle
        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=self.learning_rate)

        train_step_counter = tf.Variable(0)

        agent = dqn_agent.DqnAgent(
            train_env.time_step_spec(),
            train_env.action_spec(),
            q_network=q_net,
            optimizer=optimizer,
            td_errors_loss_fn=common.element_wise_squared_loss,
            train_step_counter=train_step_counter)

        agent.initialize()

        #sets up the agents policies
        eval_policy = agent.policy # main one for evaluation and deployment
        collect_policy = agent.collect_policy #second one used for data collection
        random_policy = random_tf_policy.RandomTFPolicy(train_env.time_step_spec(), train_env.action_spec()) # randomly select an action for each time step

        #keeps track of data environment collects
        replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
            data_spec=agent.collect_data_spec,
            batch_size=train_env.batch_size,
            max_length= self.replay_buffer_max_length)

            # Dataset generates trajectories with shape [Bx2x...] This is so that the agent has access to both the current
        # and previous state to compute loss. Parallel calls and prefetching are used to optimize process.



        dataset = replay_buffer.as_dataset(
            num_parallel_calls=3,
            sample_batch_size=self.batch_size,
            num_steps=2).prefetch(3)
        iterator = iter(dataset)

        # (Optional) Optimize by wrapping some of the code in a graph using TF function.
        agent.train = common.function(agent.train)

        # Reset the train step
        agent.train_step_counter.assign(0)

        # Evaluate the agent's policy once before training.
        avg_rewards = self.compute_avg_return(eval_env, agent.policy, self.num_eval_episodes)
        rewards = [avg_rewards]

        self.collect_data(train_env, random_policy, replay_buffer, steps=5000)
        train_env.reset()

        # We initially fill the replay buffer with 100 trajectories to help the assistant
#        collect_data(train_env, random_policy, replay_buffer, steps=5000)
    #    train_env.reset()

        # Here, we run the simulation to train the agent

        for _ in range(self.num_iterations):
            # Collect a few steps using collect_policy and save to the replay buffer.
            #we need to run this as many times as our hyperparameter specifies but because it's only once we only run it once here
            self.collect_step(train_env, agent.collect_policy, replay_buffer)

            # Sample a batch of data from the buffer and update the agent's network.
            experience, unused_info = next(iterator)
            train_loss = agent.train(experience).loss

            # Number of training steps so far
            step = agent.train_step_counter.numpy()

            # Prints every 1000 steps made by the training agent
            if step % self.log_interval == 0:
               print('Moves made = {0}'.format(step))

            # Evaluates the agent's policy every 5000 steps, prints results,
            # ands saves the results for later so they can be plotted
            if step % self.eval_interval == 0:
                avg_rewards = self.compute_avg_return(eval_env, agent.policy, self.num_eval_episodes)
                print('step = {0}: Average Return = {1}'.format(step, avg_rewards))
                rewards.append(avg_rewards)



        iterations = range(0, self.num_iterations + 1, self.eval_interval)
        plt.plot(iterations, rewards)
        plt.xlabel('Number of Steps Trained')
        plt.ylabel('Score')
        plt.title('Snake Reinforcement Learning')
        plt.show()


    def compute_avg_return(self, environment, policy, num_episodes=10):
        #metric used to evaluate the policy, average return is the sum of rewards obtained while running a policy in an environment for an episode.
        #Several episodes are run, creating an average return.

        total_return = 0.0
        for _ in range(num_episodes):

            time_step = environment.reset()
            episode_return = 0.0

            while not time_step.is_last():
                action_step = policy.action(time_step)
                time_step = environment.step(action_step.action)
                episode_return += time_step.reward
            total_return += episode_return

        avg_return = total_return / num_episodes
        return avg_return.numpy()[0]

    def collect_step(self, environment, policy, buffer):
        time_step = environment.current_time_step()
        action_step = policy.action(time_step)
        next_time_step = environment.step(action_step.action)
        traj = trajectory.from_transition(time_step, action_step, next_time_step)

        # Add trajectory to the replay buffer
        buffer.add_batch(traj)

    def collect_data(self, env, policy, buffer, steps):
        for _ in range(steps):
            self.collect_step(env, policy, buffer)
