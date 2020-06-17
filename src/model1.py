# -*- coding: utf-8 -*-
"""
Created on Fri Nov 23 09:57:15 2018

@author: Muhammed Shifas Pv
University of Crete (UoC)
"""
from __future__ import division
import os
import sys
import math
import logging
import time
import numpy as np
import tensorflow as tf
from datetime import datetime
from lib.ops import conv
from lib.util import l1_l2_loss
from lib.precision import _FLOATX
from lib.model_io import save_variables, get_info
from lib.util import compute_receptive_field_length
import pdb

def get_var_maybe_avg(var_name, ema, **kwargs):
    ''' utility for retrieving polyak averaged params '''
    v = tf.get_variable(var_name, **kwargs)
    if ema is not None:
        v = ema.average(v)
    return v

def get_weight_variable(name, shape=None, initializer=tf.contrib.layers.xavier_initializer_conv2d(), ema=None):
    if shape is None:
        return get_var_maybe_avg(name, ema)
    else:  
        return get_var_maybe_avg(name, ema, shape=shape, dtype=_FLOATX, initializer=initializer)

def get_bias_variable(name, shape=None, initializer=tf.constant_initializer(value=0.0, dtype=_FLOATX), ema=None): 
    if shape is None:
        return get_var_maybe_avg(name, ema)
    else:  
        return get_var_maybe_avg(name, ema, shape=shape, dtype=_FLOATX, initializer=initializer)
   
def get_getter(ema):
    def ema_getter(getter, name, *args, **kwargs):
        var = getter(name, *args, **kwargs)
        ema_var = ema.average(var)
        return ema_var if ema_var else var
    return ema_getter




class FFTNEt(object):
    
    def __init__(self,cfg,model_id=None):
        self.cfg=cfg
        self.n_channels = cfg['n_channels'] 
#        self.network_depth=cfg['network_depth']
        self.use_biases = cfg['use_biases']
        self.l2 = cfg['L2_regularization'] 
        self.use_dropout = cfg['use_dropout']
        self.use_ema = cfg['use_ema']
        self.polyak_decay = cfg['polyak_decay']
        self.dilations=cfg['dilations']
        self.model_id = model_id    
        self.target_length= cfg['target_length']
        self.filter_length=cfg[ 'filter_length']
        self.input_length=cfg['input_length']
        self.receptive_field=compute_receptive_field_length(self.dilations)
        self.half_receptive_field=int(self.receptive_field//2)
        
        self.create_variables()
        if self.use_ema:
            self.ema = tf.train.ExponentialMovingAverage(decay=self.polyak_decay)
        else:
            self.ema = None 
    
        
    def create_variables(self):       
        r = self.n_channels
        fw = self.filter_length
        with tf.variable_scope('FFTNet'):
            with tf.name_scope('causal_layer'):
                get_weight_variable('W', (1, fw, 1, r))  # filter_height, filter_width, in_channels, out_channels
                if self.use_biases['causal_layer']:
                    get_bias_variable('b', (r))     
                    

                
            with tf.variable_scope('FFT_layers_Conv1D'): # implementing (Conv1D_L+ Conv1D_R) as a single filter convolution
                for i,feature_dim in enumerate(self.dilations):
                    with tf.variable_scope('block{}'.format(i)):
                        get_weight_variable('Conv1D', (1, fw, r, r)) # filter_height, filter_width, in_channels, out_channels                         
                        if self.use_biases['bias']:
                            get_bias_variable('bias', (r))
                                                                
                        get_weight_variable('Conv1D_block',(1,1,r,r)) # Block output 1D convolution 
                        if self.use_biases['bias_block']:
                            get_bias_variable('bias_block', (r)) 
                                


            with tf.name_scope('postprocessing'):
                # final Fully connected layer
                get_weight_variable('FC_layer', (1, 1, r, 1))  
                if self.use_biases['FC_layer']:
                    get_bias_variable('FC_layer', (1))                           

 
    def postprocessing(self, X, ema=None):
        # post processing is a Fully connected network (FC)
        r = self.n_channels
        p1 = 1
        with tf.name_scope('postprocessing'):

            W = get_weight_variable('FC_layer', shape=(1, 1, r, 1), ema=ema)
            X = conv(X, W) 
            if self.use_biases['FC_layer']:
                b = get_bias_variable('FC_layer', shape=(p1), ema=ema)
                X += b
        return X
      
        
    def causal_layer(self, X, ema=None):
        fw = self.filter_length
        r = self.n_channels

        with tf.name_scope('causal_layer'):
            W = get_weight_variable('W', shape=(1, fw, 1, r), ema=ema) # filter_height, filter_width, in_channels, out_channels   
            Y = conv(X, W)
            if self.use_biases['causal_layer']:
                b = get_bias_variable('b', shape=(r), ema=ema) 
                Y += b
 
            Y = tf.tanh(Y) 

        return Y
    
    
    
    def FFT_layer(self,layer_input, index, dilation, is_training=True, ema=None):
        
        fw = self.filter_length 
        r = self.n_channels
        
        with tf.variable_scope('FFT_layers_Conv1D'):
            with tf.variable_scope('block{}'.format(index)):
                
                W = get_weight_variable('Conv1D', shape=(1, fw, r, r), ema=ema) 
                Y = conv(layer_input, W,dilation)
                if self.use_biases['bias']:                                         # Block (conv1D_L+conv1D_R)
                    b=get_bias_variable('bias', (r))    
                    Y +=b
                                                     
                Z=tf.nn.relu(Y) 
                W_block=get_weight_variable('Conv1D_block',(1,1,r,r)) 
                block_output=conv(Z,W_block)                                         # Block output Relu->Conv1D->Relu
                X=tf.nn.relu(block_output)   
                Z=X[:,:,dilation:-dilation,:] +layer_input[:,:,dilation:-dilation,:]     
        return Z
        
    
    def get_out_1_loss(self, Y_true, Y_pred):

        weight = self.cfg['loss']['out_1']['weight']
        l1_weight = self.cfg['loss']['out_1']['l1']
        l2_weight = self.cfg['loss']['out_1']['l2']


        if weight == 0:
            return Y_true * 0

        return weight * l1_l2_loss(Y_true, Y_pred, l1_weight, l2_weight)
    
    
    
    def inference(self, X, is_training, ema): 

        with tf.variable_scope('FFTNet', reuse=True):
            #X ->Causal_layer -> FFT_layer0-> ... FFT_layerN
            X = self.causal_layer(X, ema) 

            # FFT Layers  
            for i, dilation in enumerate(self.dilations):         
                X= self.FFT_layer(X, i, dilation, is_training, ema)
            # post processing
            clean_audio_pred= self.postprocessing(X, ema)
        return clean_audio_pred        
        
        
        
    def define_train_computations(self, optimizer, train_audio_conditions_reader, valid_audio_conditions_reader, global_step):

        # Train operations 
        self.train_audio_conditions_reader = train_audio_conditions_reader

        mixed_audio_train, clean_audio_train= train_audio_conditions_reader.dequeue()
        
        clean_audio_train = clean_audio_train[:, :, self.half_receptive_field:-self.half_receptive_field, :]  # target1

        clean_audio_pred = self.inference(mixed_audio_train, is_training=True, ema=None)
        self.train_loss = self.get_out_1_loss(clean_audio_train, clean_audio_pred)
        trainable_variables = tf.trainable_variables()


#        # Regularization loss 
        if self.l2 is not None:
            l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if not('_b' in v.name)])
            self.train_loss += self.l2*l2_loss

        trainable_variables = tf.trainable_variables()
        self.gradients_update_op = optimizer.minimize(self.train_loss, global_step=global_step, var_list=trainable_variables)
        if self.use_ema:
            self.maintain_averages_op = tf.group(self.ema.apply(trainable_variables)) 
            self.update_op = tf.group(self.gradients_update_op, self.maintain_averages_op)
        else:
            self.update_op = self.gradients_update_op

         

        # Validation operations
        self.valid_audio_conditions_reader = valid_audio_conditions_reader

        mixed_audio_valid, clean_audio_valid = valid_audio_conditions_reader.dequeue()

        clean_audio_valid = clean_audio_valid[:, :, self.half_receptive_field:-self.half_receptive_field, :]   # target 1

        clean_audio_pred_valid = self.inference(mixed_audio_valid, is_training=False, ema=self.ema)
     

        # Loss of validation data
        self.valid_loss = self.get_out_1_loss(clean_audio_valid, clean_audio_pred_valid)        
  
      
    def train_epoch(self, coord, sess, logger):
        self.train_audio_conditions_reader.reset()
        thread = self.train_audio_conditions_reader.start_enqueue_thread(sess) 

        total_train_loss = 0
        total_batches = 0 
        
        while (not coord.should_stop()) and self.train_audio_conditions_reader.check_for_elements_and_increment():
            batch_loss, _ = sess.run([self.train_loss, self.update_op]) 
            if math.isnan(batch_loss):
                logger.critical('train cost is NaN')
                coord.request_stop() 
                break 
            total_train_loss += batch_loss
            total_batches += 1  
        
        coord.join([thread])
        
        if total_batches > 0:  
            average_train_loss = total_train_loss/total_batches  

        return average_train_loss         
        
    
    def valid_epoch(self, coord, sess, logger):
        self.valid_audio_conditions_reader.reset()
        thread = self.valid_audio_conditions_reader.start_enqueue_thread(sess) 

        total_valid_loss = 0
        total_batches = 0 

        while (not coord.should_stop()) and self.valid_audio_conditions_reader.check_for_elements_and_increment():
            batch_loss = sess.run(self.valid_loss)
            if math.isnan(batch_loss):
                logger.critical('valid cost is NaN')
                coord.request_stop()
                break  
            total_valid_loss += batch_loss
            total_batches += 1  

        coord.join([thread])  

        if total_batches > 0:  
            average_valid_loss = total_valid_loss/total_batches  

        return average_valid_loss        
        
        
    def train(self, cfg, coord, sess):
        logger = logging.getLogger("msg_logger") 

        started_datestring = "{0:%Y-%m-%d, %H-%M-%S}".format(datetime.now())
        logger.info('Training of FFTNet started at: ' + started_datestring + ' using Tensorflow.\n')
        logger.info(get_info(cfg))

#        if self.use_batch_normalization and self.use_biases['filter_gate']:
#            print('Warning: Batch normalization should not be used in combination with filter and gate biases.')
#            logger.warning('Warning: Batch normalization should not be used in combination with filter and gate biases. Change the configuration file.')

        start_time = time.time()

        n_early_stop_epochs = cfg['n_early_stop_epochs']
        n_epochs = cfg['n_epochs']
       
        saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=4)

        early_stop_counter = 0

        min_valid_loss = sys.float_info.max
        epoch = 0
        while (not coord.should_stop()) and (epoch < n_epochs):
            epoch += 1
            epoch_start_time = time.time() 
            train_loss = self.train_epoch(coord, sess, logger) 
            valid_loss = self.valid_epoch(coord, sess, logger) 

            epoch_end_time = time.time()
                         
            info_str = 'Epoch=' + str(epoch) + ', Train: ' + str(train_loss) + ', Valid: '
            info_str += str(valid_loss) + ', Time=' + str(epoch_end_time - epoch_start_time)  
            logger.info(info_str)

            if valid_loss < min_valid_loss: 
                logger.info('Best epoch=' + str(epoch)) 
                save_variables(sess, saver, epoch, cfg, self.model_id) 
                min_valid_loss = valid_loss 
                early_stop_counter = 0
            else:
                early_stop_counter += 1

            if early_stop_counter > n_early_stop_epochs:
                # too many consecutive epochs without surpassing the best model
                logger.debug('stopping early')
                break

        end_time = time.time()
        logger.info('Total time = ' + str(end_time - start_time))

        if (not coord.should_stop()):
            coord.request_stop()        
        
    def define_generation_computations(self): 
        self.noisy_audio_test = tf.placeholder(shape=(1, 1, None, 1), dtype=_FLOATX)
     
#       
        if self.use_ema:
            self.inference(self.noisy_audio_test, self.lc_test, self.gc_test, is_training=False, ema=None)
            self.ema.apply(tf.trainable_variables()) 

        self.clean_audio_pred_computation_graph = self.inference(self.noisy_audio_test, is_training=False, ema=self.ema) 

        
        

    def generation(self, sess, noisy_audio, lc=None, gc=None):
        n_samples = noisy_audio.shape[0] 
        noisy_audio=np.append(np.random.normal(0,0.001,3069), noisy_audio)
        noisy_audio=np.append(noisy_audio,np.zeros((3069,)))

        noisy_audio_reshaped = noisy_audio.reshape((1, 1, -1, 1)) 
        
        feed_dict = {self.noisy_audio_test:noisy_audio_reshaped}

        start_time = time.time()
        clean_audio_pred = sess.run(self.clean_audio_pred_computation_graph, feed_dict=feed_dict)
        end_time = time.time()
        print('Audio Duration = ' + str(n_samples/16000))
        print('Total processing time = ' + str(end_time - start_time))
        clean_audio_pred = clean_audio_pred.reshape((-1, ))

        noise_pred = noisy_audio[self.half_receptive_field:-self.half_receptive_field] - clean_audio_pred   

        return clean_audio_pred, noisy_audio
      
        
        
        
        
        
        
        
        
        
        
        
        
        
