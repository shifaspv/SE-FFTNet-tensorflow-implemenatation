# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 11:24:33 2020

@author: shifas
"""

import os

directory='./NSDTSEA/noisy_trainset_wav' # path to the audio directory
arr = os.listdir(directory)
arr.sort()
print(arr)
f=open('./NSDTSEA/train_id_list.txt','w') # name of the list file

for id in arr:
    f.write(id[0:-4]+'\n')

f.close()
