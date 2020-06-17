import tensorflow as tf
import numpy as np


def get_learning_rate(name, global_step, params):

     name = name.lower()

     if name == 'constant':
        return params['learning_rate']
     elif name == 'exponential':
        # decayed_learning_rate = learning_rate * decay_rate ^ (global_step / decay_steps)
        return tf.train.exponential_decay(global_step=global_step, **params)
     elif name == 'natural_exp':
        # decayed_learning_rate = learning_rate * exp(-decay_rate * global_step) 
        return tf.train.natural_exp_decay(global_step=global_step, **params) 
     elif name == 'inverse_time':
        # decayed_learning_rate = learning_rate / (1 + decay_rate * t)
        return inverse_time_decay(global_step=global_step, **params)
     elif name == 'piecewise_constant':
        return tf.train.piecewise_constant(global_step=global_step, **params) 
     elif name == 'polynomial':
        # if cylce == False:
        #    global_step = min(global_step, decay_steps)
        # if cycle == True:
        #    decay_steps = decay_steps * ceil(global_step / decay_steps)
        # decayed_learning_rate = (learning_rate - end_learning_rate) *
        #            (1 - global_step / decay_steps) ^ (power) + end_learning_rate
        return tf.train.polynomial_decay(global_step=global_step, **params) 
     else:
        raise Exception('The learning rate method ' + name + ' has not been implemented.')
         

def get_optimizer(name, learning_rate, params):

    name = name.lower()

    if name == 'sgd':
        return tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    elif name == 'momentum':  
        return tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9)    
    elif name == 'adadelta':
        # learning_rate=0.001, rho=0.95, epsilon=1e-08
        return tf.train.AdadeltaOptimizer(learning_rate=learning_rate, **params)
    elif name == 'adagrad':
        # initial_accumulator_value=0.1
        return tf.train.AdagradOptimizer(learning_rate=learning_rate, **params) 
    elif name == 'rmsprop':
        # decay=0.9, momentum=0.0, epsilon=1e-10, centered=False,
        return tf.train.RMSPropOptimizer(learning_rate=learning_rate, **params)
    elif name == 'adam':
        # learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-08
        return tf.train.AdamOptimizer(learning_rate=learning_rate, **params)     
    else:
        raise Exception('The optimization method ' + name + ' has not been implemented.')
          
    


