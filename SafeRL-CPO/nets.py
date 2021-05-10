from sklearn.utils import shuffle
from collections import deque
from copy import deepcopy
import tensorflow as tf
import numpy as np
import itertools
import pickle
import random
import copy
import time
import os

EPS = 1e-10

class Agent:
    def __init__(self, env, args):
        self.env = env
        self.name = args['agent_name']
        self.checkpoint_dir='{}/checkpoint'.format(args['env_name'])
        self.discount_factor = args['discount_factor']
        self.state_dim = env.observation_space.shape[0]
        try:
            self.action_dim = env.action_space.shape[0]
            self.action_bound_min = env.action_space.low
            self.action_bound_max = env.action_space.high
        except:
            self.action_dim = 1
            self.action_bound_min = 0.0
            self.action_bound_max = 4.0
            # self.action_bound_min = - 1.0
            # self.action_bound_max = 1.0
        self.hidden1_units = args['hidden1']
        self.hidden2_units = args['hidden2']
        self.v_lr = args['v_lr']
        self.cost_v_lr = args['cost_v_lr']
        self.value_epochs = args['value_epochs']
        self.cost_value_epochs = args['cost_value_epochs']
        self.num_conjugate = args['num_conjugate']
        self.max_decay_num = args['max_decay_num']
        self.line_decay = args['line_decay']
        self.max_kl = args['max_kl']
        self.max_avg_cost = args['max_avg_cost']
        self.damping_coeff = args['damping_coeff']
        self.gae_coeff = args['gae_coeff']

        with tf.variable_scope(self.name):
            #placeholder
            self.states = tf.placeholder(tf.float32, [None, self.state_dim*5], name='State')
            self.actions = tf.placeholder(tf.float32, [None, self.action_dim], name='Action')
            self.targets = tf.placeholder(tf.float32, [None,], name='targets')
            self.cost_targets = tf.placeholder(tf.float32, [None,], name='cost_targets')
            self.gaes = tf.placeholder(tf.float32, [None,], name='gaes')
            self.cost_gaes = tf.placeholder(tf.float32, [None,], name='cost_gaes')
            self.old_mean = tf.placeholder(tf.float32, [None, self.action_dim], name='old_mean')
            self.old_std = tf.placeholder(tf.float32, [None, self.action_dim], name='old_std')
            self.log_prob_old = tf.placeholder(tf.float32, [None,], name='log_prob_old')

            #policy & value
            self.mean, self.std = self.build_policy_model('policy')
            self.kl = self.get_kl_divergence()
            self.value = self.build_value_model('value')
            self.cost_value = self.build_value_model('cost_value')

            #action
            self.norm_noise_action = self.mean + tf.multiply(tf.random_normal(tf.shape(self.mean)), self.std)
            self.sample_noise_action = self.unnormalize_action(self.norm_noise_action)
            self.sample_action = self.unnormalize_action(self.mean)
            norm_actions = self.normalize_action(self.actions)

            #value loss
            v_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=tf.get_variable_scope().name+'/value')
            self.v_loss = 0.5*tf.square(self.targets - self.value)
            self.v_loss = tf.reduce_mean(self.v_loss)
            v_optimizer = tf.train.AdamOptimizer(learning_rate=self.v_lr)
            self.v_gradients = tf.gradients(self.v_loss, v_vars)
            self.v_train_op = v_optimizer.apply_gradients(zip(self.v_gradients, v_vars))

            #cost_value loss
            cost_v_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=tf.get_variable_scope().name+'/cost_value')
            self.cost_v_loss = 0.5*tf.square(self.cost_targets - self.cost_value)
            self.cost_v_loss = tf.reduce_mean(self.cost_v_loss)
            cost_v_optimizer = tf.train.AdamOptimizer(learning_rate=self.cost_v_lr)
            self.cost_v_gradients = tf.gradients(self.cost_v_loss, cost_v_vars)
            self.cost_v_train_op = cost_v_optimizer.apply_gradients(zip(self.cost_v_gradients, cost_v_vars))

            #policy optimizer
            p_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=tf.get_variable_scope().name+'/policy')
            self.log_prob = - tf.reduce_sum(tf.log(self.std + EPS) + 0.5*np.log(2*np.pi) + tf.squared_difference(norm_actions, self.mean) / (2 * tf.square(self.std) + EPS), axis=1)
            self.objective = tf.reduce_mean(tf.exp(self.log_prob - self.log_prob_old) * self.gaes)
            self.cost_surrogate = tf.reduce_mean(tf.exp(self.log_prob - self.log_prob_old) * self.cost_gaes)
            self.grad_g = tf.gradients(self.objective, p_vars)
            self.grad_g = tf.concat([tf.reshape(g, [-1]) for g in self.grad_g], axis=0)
            self.grad_b = tf.gradients(-self.cost_surrogate, p_vars)
            self.grad_b = tf.concat([tf.reshape(b, [-1]) for b in self.grad_b], axis=0)

            kl_grad = tf.gradients(self.kl, p_vars)
            kl_grad = tf.concat([tf.reshape(g, [-1]) for g in kl_grad], axis=0)
            self.theta_ph = tf.placeholder(tf.float32, shape=kl_grad.shape, name='theta')
            self.hessian_product = tf.gradients(tf.reduce_sum(kl_grad*self.theta_ph), p_vars)
            self.hessian_product = tf.concat([tf.reshape(g, [-1]) for g in self.hessian_product], axis=0)
            self.hessian_product += self.damping_coeff * self.theta_ph

            self.flatten_p_vars = tf.concat([tf.reshape(g, [-1]) for g in p_vars], axis=0)
            self.params = tf.placeholder(tf.float32, self.flatten_p_vars.shape, name='params')
            self.assign_op = []
            start = 0
            for p_var in p_vars:
                size = np.prod(p_var.shape)
                param = tf.reshape(self.params[start:start + size], p_var.shape)
                self.assign_op.append(p_var.assign(param))
                start += size

            #make session and load model
            self.sess = tf.Session()
            self.sess.run(tf.global_variables_initializer())
            # self.load()


    def Hx(self, theta, feed_inputs):
        feed_inputs[self.theta_ph] = theta
        return self.sess.run(self.hessian_product, feed_dict=feed_inputs)

    def get_kl_divergence(self):
        mean, std = self.mean, self.std
        old_mean, old_std = self.old_mean, self.old_std
        log_std_old = tf.log(old_std + EPS)
        log_std_new = tf.log(std + EPS)
        frac_std_old_new = old_std/(std + EPS)
        kl = tf.reduce_mean(tf.reduce_sum(log_std_new - log_std_old + 0.5*tf.square(frac_std_old_new) + 0.5*tf.square((mean - old_mean)/(std + EPS))- 0.5, axis=1))
        return kl

    def build_policy_model(self, name='policy', reuse=False):
        param_initializer = lambda : tf.random_normal_initializer(mean=0.0, stddev=0.01)
        with tf.variable_scope(name, reuse=reuse):
            model = tf.layers.dense(self.states, self.hidden1_units, activation=None, bias_initializer=param_initializer(), kernel_initializer=param_initializer())
            model = tf.layers.batch_normalization(model)
            model = tf.nn.tanh(model)
            model = tf.layers.dense(model, self.hidden2_units, activation=None, bias_initializer=param_initializer(), kernel_initializer=param_initializer())
            model = tf.layers.batch_normalization(model)
            model = tf.nn.tanh(model)
            mean = tf.layers.dense(model, self.action_dim, activation=tf.tanh, bias_initializer=param_initializer(), kernel_initializer=param_initializer())
            logits_std = tf.get_variable("logits_std",shape=(self.action_dim), initializer=tf.random_normal_initializer(mean=-1.0,stddev=0.01))
            std = tf.ones_like(mean)*tf.nn.softplus(logits_std)
        return mean, std

    def build_value_model(self, name='value', reuse=False):
        param_initializer = lambda : tf.random_normal_initializer(mean=0.0, stddev=0.01)
        with tf.variable_scope(name, reuse=reuse):
            inputs = self.states
            model = tf.layers.dense(inputs, self.hidden1_units, activation=None, bias_initializer=param_initializer(), kernel_initializer=param_initializer())
            model = tf.layers.batch_normalization(model)
            model = tf.nn.tanh(model)
            model = tf.layers.dense(model, self.hidden2_units, activation=None, bias_initializer=param_initializer(), kernel_initializer=param_initializer())
            model = tf.layers.batch_normalization(model)
            model = tf.nn.tanh(model)
            model = tf.layers.dense(model, 1, activation=None, bias_initializer=param_initializer(), kernel_initializer=param_initializer())
            model = tf.reshape(model, [-1])
            return model

    def normalize_action(self, a):
        temp_a = 2.0/(self.action_bound_max - self.action_bound_min)
        temp_b = (self.action_bound_max + self.action_bound_min)/(self.action_bound_min - self.action_bound_max)
        temp_a = tf.ones_like(a)*temp_a
        temp_b = tf.ones_like(a)*temp_b
        return temp_a*a + temp_b

    def unnormalize_action(self, a):
        temp_a = (self.action_bound_max - self.action_bound_min)/2.0
        temp_b = (self.action_bound_max + self.action_bound_min)/2.0
        temp_a = tf.ones_like(a)*temp_a
        temp_b = tf.ones_like(a)*temp_b
        return temp_a*a + temp_b

    def get_action(self, state, is_train):
        if is_train:
            [[action], [value], [cost_value]] = self.sess.run([self.sample_noise_action, self.value, self.cost_value], feed_dict={self.states:[state]})
        else:
            [[action], [value], [cost_value]] = self.sess.run([self.sample_action, self.value, self.cost_value], feed_dict={self.states:[state]})
        clipped_action = np.clip(np.rint(action), self.action_bound_min, self.action_bound_max).astype(int)[0]

        return action, clipped_action, value, cost_value

    def get_gaes_targets(self, rewards, values, next_values):
        deltas = np.array(rewards) + self.discount_factor*np.array(next_values) - np.array(values)
        gaes = deepcopy(deltas)
        for t in reversed(range(len(gaes))):
            if t < len(gaes) - 1:
                gaes[t] = gaes[t] + self.discount_factor*self.gae_coeff*gaes[t + 1]
        targets = np.array(values) + np.array(gaes)
        return gaes, targets

    def train(self, trajs):
        states = trajs[0]
        actions = trajs[1]
        targets = trajs[2]
        cost_targets = trajs[3]
        gaes = trajs[4]
        cost_gaes = trajs[5]
        avg_costs = trajs[6]
        old_means, old_stds, old_log_probs = self.sess.run([self.mean, self.std, self.log_prob], 
                                                    feed_dict={self.states:states, self.actions:actions})

        #POLICY update
        grad_g, grad_b = self.sess.run([self.grad_g, self.grad_b], feed_dict={
            self.states:states,
            self.actions:actions,
            self.log_prob_old:old_log_probs,
            self.gaes:gaes,
            self.cost_gaes:cost_gaes})
        feed_inputs = {self.states:states, self.old_mean:old_means, self.old_std:old_stds}
        x_value = self.conjugate_gradient_method(grad_g, feed_inputs)
        approx_g = self.Hx(x_value, feed_inputs)

        #############################################
        ####### solve lagrangian dual problem #######
        c_value = np.mean(avg_costs) - self.max_avg_cost
        print("constraint value :", c_value)
        if np.dot(grad_b, grad_b) <= 1e-8 and c_value < 0:
            H_inv_b, scalar_r, scalar_s, A_value, B_value = 0, 0, 0, 0, 0
            scalar_q = np.inner(approx_g, x_value)
            optim_case = 4
        else:
            H_inv_b = self.conjugate_gradient_method(grad_b, feed_inputs)
            approx_b = self.Hx(H_inv_b, feed_inputs)
            scalar_q = np.inner(approx_g, x_value)
            scalar_r = np.inner(approx_g, H_inv_b)
            scalar_s = np.inner(approx_b, H_inv_b)
            A_value = scalar_q - scalar_r**2 / scalar_s    # should be always positive (Cauchy-Shwarz)
            B_value = 2*self.max_kl - c_value**2 / scalar_s  # does safety boundary intersect trust region? (positive = yes)
            print("B value :", B_value)
            if c_value < 0 and B_value < 0:
                # point in trust region is feasible and safety boundary doesn't intersect
                # ==> entire trust region is feasible
                optim_case = 3
            elif c_value < 0 and B_value >= 0:
                # x = 0 is feasible and safety boundary intersects
                # ==> most of trust region is feasible
                optim_case = 2
            elif c_value >= 0 and B_value >= 0:
                # x = 0 is infeasible and safety boundary intersects
                # ==> part of trust region is feasible, recovery possible
                optim_case = 1
            else:
                # x = 0 infeasible, and safety halfspace is outside trust region
                # ==> whole trust region is infeasible, try to fail gracefully
                optim_case = 0

        print("optimizing case :", optim_case)
        if optim_case in [3,4]:
            lam = np.sqrt(scalar_q / (2*self.max_kl))
            nu = 0
        elif optim_case in [1,2]:
            LA, LB = [0, scalar_r/c_value], [scalar_r/c_value, np.inf]
            LA, LB = (LA, LB) if c_value < 0 else (LB, LA)
            proj = lambda x, L : max(L[0], min(L[1], x))
            lam_a = proj(np.sqrt(A_value/B_value), LA)
            lam_b = proj(np.sqrt(scalar_q/(2*self.max_kl)), LB)
            f_a = lambda lam : -0.5 * (A_value / (lam + EPS) + B_value * lam) - scalar_r*c_value/(scalar_s + EPS)
            f_b = lambda lam : -0.5 * (scalar_q / (lam + EPS) + 2*self.max_kl*lam)
            lam = lam_a if f_a(lam_a) >= f_b(lam_b) else lam_b
            nu = max(0, lam * c_value - scalar_r) / (scalar_s + EPS)
        else:
            lam = 0
            nu = np.sqrt(2*self.max_kl / (scalar_s+EPS))
        ####### solve lagrangian dual problem #######
        #############################################

        #line search
        delta_theta = (1./(lam + EPS))*(x_value + nu*H_inv_b) if optim_case > 0 else nu*H_inv_b
        init_theta = self.sess.run(self.flatten_p_vars)
        beta = 1.0

        init_objective, init_cost_surrogate = self.sess.run([self.objective, self.cost_surrogate], feed_dict={
            self.states:states,
            self.actions:actions,
            self.log_prob_old:old_log_probs,
            self.gaes:gaes,
            self.cost_gaes:cost_gaes})

        while True:
            #theta = beta*x_value + init_theta
            theta = beta*delta_theta + init_theta
            self.sess.run(self.assign_op, feed_dict={self.params:theta})
            kl, objective, cost_surrogate = self.sess.run([self.kl, self.objective, self.cost_surrogate], feed_dict={
                self.states:states,
                self.actions:actions,
                self.gaes:gaes,
                self.cost_gaes:cost_gaes,
                self.log_prob_old:old_log_probs,
                self.old_mean:old_means,
                self.old_std:old_stds})
            if kl <= self.max_kl and (objective > init_objective if optim_case > 1 else True) and cost_surrogate - init_cost_surrogate <= max(-c_value, 0):
                break
            beta *= self.line_decay

        #VALUE update
        v_s, v_t = shuffle(states, targets, random_state=0)
        for _ in range(self.value_epochs):
            v_s, v_t = shuffle(v_s, v_t, random_state=0)
            self.sess.run([self.v_train_op], feed_dict={
                    self.states:v_s,
                    self.targets:v_t})
        v_loss = self.sess.run(self.v_loss, feed_dict={self.states:states, self.targets:targets})

        #COST VALUE update
        v_s, v_t = shuffle(states, cost_targets, random_state=0)
        for _ in range(self.cost_value_epochs):
            v_s, v_t = shuffle(v_s, v_t, random_state=0)
            self.sess.run([self.cost_v_train_op], feed_dict={
                    self.states:v_s, 
                    self.cost_targets:v_t})
        cost_v_loss = self.sess.run(self.cost_v_loss, feed_dict={self.states:states, self.cost_targets:cost_targets})

        return v_loss, cost_v_loss, objective, cost_surrogate, kl

    def conjugate_gradient_method(self, g, feed_inputs):
        x_value = np.zeros_like(g)
        residue = deepcopy(g)
        p_vector = deepcopy(g)
        rs_old = np.inner(residue, residue)
        for i in range(self.num_conjugate):
            Ap = self.Hx(p_vector, feed_inputs)
            pAp = np.inner(p_vector, Ap)
            alpha = rs_old / (pAp + EPS)
            x_value += alpha * p_vector
            residue -= alpha * Ap
            rs_new = np.inner(residue, residue)
            p_vector = residue + (rs_new / rs_old) * p_vector
            rs_old = rs_new
        return x_value

    def save(self):
        
        self.saver = tf.train.Saver(var_list= tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=tf.get_variable_scope().name))
        if not os.path.isdir(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        self.saver.save(self.sess, self.checkpoint_dir+'/model.ckpt')
        print('save 성공!')

    def load(self):
        self.saver = tf.train.Saver(var_list= tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=tf.get_variable_scope().name))

        if not os.path.isdir(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        ckpt = tf.train.get_checkpoint_state(self.checkpoint_dir)
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
            print('success to load model!', ckpt.model_checkpoint_path)
        else:
            self.sess.run(tf.global_variables_initializer())
            print('fail to load model...')
        

# if __name__ == "__main__":
#     env_name = 'Pendulum-v0'
#     save_name = env_name.split('-')[0]
#     agent_args = {'agent_name':'TRPO',
#                 'env_name':save_name,
#                 'discount_factor':0.9,
#                 'hidden1':2,
#                 'hidden2':2,
#                 'v_lr':1e-3,
#                 'std':0.1}

#     import gym
#     env = gym.make(env_name)
#     agent = Agent(env, agent_args)

#     states = []
#     actions = []
#     rewards = []
#     next_states = []
#     state = env.reset()
#     done = False
#     while not done:
#         action = agent.get_action(state, True)
#         next_state, reward, done, info = env.step(action)
#         states.append(state)
#         actions.append(action)
#         rewards.append(reward)
#         next_states.append(next_state)
        
#         state = next_state

#     trajs = [states, actions, rewards, next_states]
#     agent.train(trajs)

