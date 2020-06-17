from __future__ import division, print_function

import os
import logging
import argparse
import json
import numpy as np
import tensorflow as tf

def get_configuration(learning_phase):
    parser = argparse.ArgumentParser(description='FFTNet argument parser')
    parser.add_argument('--config', type=str, default='config_params.json', help='The configuration filename')
    parser.add_argument('--model_id', type=int, default=None, help='The model id. It is used to recover a saved model')
    if learning_phase == 'generation':     
        parser.add_argument('--noisy_speech_filename', type=str, default='p232_010', help='The input noisy filename') 
        parser.add_argument('--output_filename', type=str, default=None, help='The generated clean and noise base filename')  
     
    args = parser.parse_args()

    if learning_phase == 'generation':

        if args.noisy_speech_filename is None:
            raise Exception('Please give the filename of the noisy speech') 


    config_filename = os.path.join('..', 'config', args.config)   
    with open(config_filename, 'r') as f:
        cfg = json.load(f)  

    cfg['config_filename'] = config_filename

    if cfg['input_length'] <= 0:
        cfg['input_length'] = None  

    if cfg['target_length'] <= 0:
        cfg['target_length'] = None 

    if learning_phase == 'generation':
        cfg['noisy_speech_filename'] = args.noisy_speech_filename   
        if args.output_filename is not None:
            cfg['output_clean_speech_filename'] = 'clean_' + args.output_filename
            cfg['output_noise_filename'] = 'noise_' + args.output_filename  
        else:
            cfg['output_clean_speech_filename'] = 'clean_' + args.noisy_speech_filename
            cfg['output_noise_filename'] = 'noise_' + args.noisy_speech_filename 

    if learning_phase == 'generation':
        cfg['test_noisy_audio_dir'] = os.path.join(cfg['data_dir'], cfg['test_noisy_audio_dir'])

    cfg['model_id'] = args.model_id

    optim_filename = os.path.join(cfg['base_dir'], 'config', cfg['optimization_parameters'])
    with open(optim_filename, 'r') as f:
        optim_params = json.load(f)

    cfg['optim_filename'] = optim_filename
             
    if learning_phase == 'generation':
            return cfg, optim_params['learning_rate_method'][cfg['learning_rate_method']], optim_params['optimization_algorithm'][cfg['optimization_algorithm']] 
           
    return cfg, optim_params['learning_rate_method'][cfg['learning_rate_method']], optim_params['optimization_algorithm'][cfg['optimization_algorithm']], None

def setup_logger(logger_name, log_file, level=logging.INFO):
    logger = logging.getLogger(logger_name)
    formatter = logging.Formatter('%(message)s')
    fileHandler = logging.FileHandler(log_file, mode='w')
    fileHandler.setFormatter(formatter)

    logger.setLevel(level)
    logger.addHandler(fileHandler)


def read_model_id(filename):
    if os.path.exists(filename):
        with open(filename, 'r') as fid:
            model_id = int(fid.read())
        fid.close()
    else:
        write_model_id(filename, 1)
        model_id = 0
        
    return model_id      


def write_model_id(filename, model_id):
    model_id_txt = str(model_id) 
    with open(filename, 'w') as fid:
        fid.write(model_id_txt)
    fid.close() 

def get_model_id(cfg):
    model_id_filename = os.path.join(cfg['base_dir'], cfg['saved_models_dir'], cfg['model_ids']) 
    model_id = read_model_id(model_id_filename) + 1 # Reserve the next model_id. If file does not exists then create it 
    write_model_id(model_id_filename, model_id) 

    return model_id 


def save_variables(sess, saver, epoch, cfg, model_id): 
    model_path = os.path.join(cfg['base_dir'], cfg['saved_models_dir'], str(model_id)) 
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    checkpoint_path = os.path.join(model_path, 'FFTNet')
    saver.save(sess, checkpoint_path, global_step=epoch)

def restore_variables(sess, cfg):
    '''variables_to_restore = {
    var.name[:-2]: var for var in tf.trainable_variables() #tf.global_variables()
        if not ('state_buffer' in var.name or 'pointer' in var.name)}
    saver = tf.train.Saver(variables_to_restore)'''
    saver = tf.train.Saver(tf.global_variables()) # new

    model_path = os.path.join(cfg['base_dir'], cfg['saved_models_dir'], str(cfg['model_id']))
    ckpt = tf.train.get_checkpoint_state(model_path)
    print(ckpt.model_checkpoint_path)  
    saver.restore(sess, ckpt.model_checkpoint_path) 

    sess.run(tf.tables_initializer()) # new

def get_info(cfg):
    info_str = 'Configuration file: ' + cfg['config_filename'] + '\n'
    fid = open(cfg['config_filename'], 'r')
    lines = fid.readlines()
    fid.close()
    for line in lines:
        info_str += line 
    info_str += '\n'   

    
    info_str += 'Optimization parameters file: ' + cfg['optim_filename'] + '\n'
     
    optim_name = cfg['optimization_algorithm']  

    info_str += optim_name + '\n'

    fid = open(cfg['optim_filename'], 'r')
    lines = fid.readlines()
    fid.close()
    for line in lines:
        info_str += line 
        
    info_str += '\n'   

    return info_str



