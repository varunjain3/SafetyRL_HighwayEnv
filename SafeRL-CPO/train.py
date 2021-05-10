# to ignore warning message
import warnings
warnings.filterwarnings("ignore")
from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False
###########################
import os
os.environ["KMP_AFFINITY"] = "none" # no affinity
from logger import Logger
from nets import Agent

from collections import deque
import tensorflow as tf
import numpy as np
import random
import pickle
import time
import sys
# import gym
from highwayenv.highway_env.envs.highway_env import HighwayEnv
# import safety_gym
from matplotlib import animation
import matplotlib.pyplot as plt
# from car_env.env import Env
#for random seed
seed = 2
np.random.seed(seed)
tf.set_random_seed(seed)
random.seed(seed)

env_name = 'cpo_harsh'

agent_name = 'CPO_harsh'
algo = '{}_{}'.format(agent_name, seed)
save_name = '_'.join(env_name.split('-')[:-1])
save_name = "{}_{}".format(save_name, algo)
agent_args = {'agent_name':agent_name,
            'env_name':save_name,
            'discount_factor':0.8,
            'hidden1':256,
            'hidden2':256,
            'v_lr':1e-3,
            'cost_v_lr':1e-3,
            'value_epochs':80,
            'cost_value_epochs':80,
            'num_conjugate':10,
            'max_decay_num':10,
            'line_decay':0.8,
            'max_kl':0.01,
            'max_avg_cost':800/1000,
            'damping_coeff':0.01,
            'gae_coeff':0.97,
            }

def train():
    global env_name, save_name, agent_args
    # env = gym.make(env_name)
    # env = Env()
    env = HighwayEnv()
    agent = Agent(env, agent_args)
    agent.load()
    v_loss_logger = Logger(save_name, 'v_loss')
    cost_v_loss_logger = Logger(save_name, 'cost_v_loss')
    kl_logger = Logger(save_name, 'kl')
    score_logger = Logger(save_name, 'score')
    cost_logger = Logger(save_name, 'cost')
    max_steps = 100
    max_ep_len = 40
    episodes = int(max_steps/max_ep_len)
    epochs = 2000
    save_freq = 2

    log_length = 10
    p_objectives = deque(maxlen=log_length)
    c_objectives = deque(maxlen=log_length)
    v_losses = deque(maxlen=log_length)
    cost_v_losses = deque(maxlen=log_length)
    kl_divergence = deque(maxlen=log_length)
    scores = deque(maxlen=log_length*episodes)
    costs = deque(maxlen=log_length*episodes)
    from tqdm import trange
    for epoch in trange(epochs):
        states = []
        actions = []
        targets = []
        cost_targets = []
        gaes = []
        cost_gaes = []
        avg_costs = []
        ep_step = 0
        while ep_step < max_steps:
            state = env.reset()
            done = False
            score = 0
            cost = 0
            step = 0
            temp_rewards = []
            temp_costs = []
            values = []
            cost_values = []
            while True:
                step += 1
                ep_step += 1
                assert env.observation_space.contains(state)
                action, clipped_action, value, cost_value = agent.get_action(state.reshape(-1), True)
                assert env.action_space.contains(clipped_action)
                next_state, reward, done, info = env.step(clipped_action)
                COST = info['cost']
                states.append(state.reshape(5,5))
                actions.append(action)
                temp_rewards.append(reward)
                temp_costs.append(COST)
                values.append(value)
                cost_values.append(cost_value)
                state = next_state
                score += reward
                cost += COST
                if done or step >= max_ep_len:
                    break
            if step >= max_ep_len:
                action, clipped_action, value, cost_value = agent.get_action(state.reshape(-1), True)
            else:
                value = 0
                cost_value = 0
            next_values = values[1:] + [value]
            temp_gaes, temp_targets = agent.get_gaes_targets(temp_rewards, values, next_values)
            next_cost_values = cost_values[1:] + [cost_value]
            temp_cost_gaes, temp_cost_targets = agent.get_gaes_targets(temp_costs, cost_values, next_cost_values)
            avg_costs.append(np.mean(temp_costs))
            targets += list(temp_targets)
            gaes += list(temp_gaes)
            cost_targets += list(temp_cost_targets)
            cost_gaes += list(temp_cost_gaes)
            score_logger.write([step, score])
            cost_logger.write([step, cost])
            scores.append(score)
            costs.append(cost)
        states = [i.reshape(-1) for i in states]
        trajs = [states, actions, targets, cost_targets, gaes, cost_gaes, avg_costs]
        v_loss, cost_v_loss, p_objective, cost_objective, kl = agent.train(trajs)
        v_loss_logger.write([ep_step, v_loss])
        cost_v_loss_logger.write([ep_step, cost_v_loss])
        kl_logger.write([ep_step, kl])
        p_objectives.append(p_objective)
        c_objectives.append(cost_objective)
        v_losses.append(v_loss)
        cost_v_losses.append(cost_v_loss)
        kl_divergence.append(kl)

        print(np.mean(scores), np.mean(costs), np.mean(v_losses), np.mean(cost_v_losses), np.mean(kl_divergence), np.mean(c_objectives),flush =True)
        if (epoch+1)%save_freq == 0:
            agent.save()
            v_loss_logger.save()
            cost_v_loss_logger.save()
            kl_logger.save()
            score_logger.save()
            cost_logger.save()

def save_frames_as_gif(frames, episode, path='./animation', filename='gym_animation'):

    #Mess with this to change frame size
    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)

    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=50)
    anim.save(path + filename + "_" + str(episode) + ".gif", writer='imagemagick', fps=60)


def test():
    global env_name, save_name, agent_args
    import os
    env = HighwayEnv()
    agent = Agent(env, agent_args)
    agent.load()
    episodes = int(1e6)
    for episode in range(episodes):
        state = env.reset()
        done = False
        score = 0
        frames = []
        while not done:
            action, clipped_action, value, cost_value = agent.get_action(state.reshape(-1), False)
            state, reward, done, info = env.step(clipped_action)
            print(reward, '\t', info.get('cost', 0))
            score += reward
            env.render()
        print("score :",score)

if len(sys.argv)== 2 and sys.argv[1] == 'test':
    test()
else:
    train()