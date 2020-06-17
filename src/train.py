# -*- coding: utf-8 -*-
"""
Created on Fri Nov 23 09:05:50 2018

@author: Muhammed Shifas PV
University of Crete (UoC)
"""
import os
import logging
import numpy as np
import tensorflow as tf
from model1 import FFTNEt
from lib.model_io import restore_variables
from lib.optimizers import get_learning_rate,get_optimizer
from lib.audio_conditions_io import AudioConditionsReader
from lib.model_io import get_configuration, setup_logger, get_model_id
from lib.util import compute_receptive_field_length
import pdb
cfg, learning_rate_params, optim_params, gc = get_configuration('train')
os.environ["CUDA_VISIBLE_DEVICES"] = cfg["CUDA_VISIBLE_DEVICES"]

if cfg['model_id'] is not None:
    model_id = cfg['model_id']
else:
    model_id = get_model_id(cfg)

msg_logging_dir = os.path.join(cfg['base_dir'], cfg['logging_dir'], 'log_'+str(model_id)+'.txt') 
setup_logger('msg_logger', msg_logging_dir, level=logging.INFO)
warning_logging_dir = os.path.join(cfg['base_dir'], cfg['logging_dir'], 'warning_'+str(model_id)+'.txt') 
setup_logger('warning_logger', warning_logging_dir, level=logging.WARNING)

receptive_field=compute_receptive_field_length(cfg['dilations'])
#receptive_field=cfg[]
coord = tf.train.Coordinator()

with tf.name_scope('create_readers'):
                 
    train_file_list = os.path.join(cfg['data_dir'], cfg['train_file_list']) 
    train_clean_audio_dir = os.path.join(cfg['data_dir'], cfg['train_clean_audio_dir'])
    train_noisy_audio_dir = os.path.join(cfg['data_dir'], cfg['train_noisy_audio_dir'])
    train_audio_reader = AudioConditionsReader(coord, train_file_list, train_clean_audio_dir, train_noisy_audio_dir, 
                                     cfg['audio_ext'],cfg['sample_rate'],cfg['zero_pad_percentage'], cfg['regain'], batch_size=cfg['batch_size'], input_length=None, 
                                     receptive_field=receptive_field, target_length=cfg['target_length'],silence_threshold=cfg['silence_threshold'], queue_size=cfg['queue_size'], 
                                     permute_segments=cfg['permute_segments'])

    valid_file_list = os.path.join(cfg['data_dir'], cfg['valid_file_list']) 
    valid_clean_audio_dir = os.path.join(cfg['data_dir'], cfg['valid_clean_audio_dir'])
    valid_noisy_audio_dir = os.path.join(cfg['data_dir'], cfg['valid_noisy_audio_dir']) 
    valid_audio_reader = AudioConditionsReader(coord, valid_file_list, valid_clean_audio_dir, valid_noisy_audio_dir, 
                                     cfg['audio_ext'], cfg['sample_rate'], cfg['zero_pad_percentage'], cfg['regain'], batch_size=cfg['batch_size'], input_length=None,
                                     receptive_field=receptive_field, target_length=cfg['target_length'], silence_threshold=cfg['silence_threshold'], queue_size=cfg['queue_size'], 
                                     permute_segments=cfg['permute_segments'])

# define learning rate decay method 
global_step = tf.Variable(0, trainable=False, name='global_step')
learning_rate = get_learning_rate(cfg['learning_rate_method'], global_step, learning_rate_params)
# define the optimization algorithm
opt_name = cfg['optimization_algorithm'].lower()
optimizer = get_optimizer(opt_name, learning_rate, optim_params)


# Create the network
FFTNet = FFTNEt(cfg, model_id)

# Define the train computation graph
FFTNet.define_train_computations(optimizer, train_audio_reader, valid_audio_reader, global_step)

sess = tf.Session()
init_op = tf.global_variables_initializer()  # New Tsiaras    
sess.run(init_op) 
# Recover the parameters of the model
if cfg['model_id'] is not None:
    print('Restore the parameters of model ' + str(cfg['model_id']))
    restore_variables(sess, cfg)
else:
    print('Train new model') 


try:
    FFTNet.train(cfg, coord, sess)
except KeyboardInterrupt:  
    print()
finally:
    if not coord.should_stop():
        coord.request_stop() 
    sess.close() 

