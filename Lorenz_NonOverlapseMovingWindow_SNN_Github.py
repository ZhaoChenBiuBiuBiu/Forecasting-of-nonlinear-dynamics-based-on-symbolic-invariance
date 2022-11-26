# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 21:01:33 2020

@author: FromM
"""
import torch
import time
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import scipy.io
from Utils_NonOverlapseMovingWindow_SNN_Github import *
import copy
import os

start_time = time.time()

if torch.cuda.is_available():
    cuda_tag = "cuda:1"
    device = torch.device(cuda_tag)  # ".to(device)" should be added to all models and inputs
    print("Running on " + cuda_tag)
else:
    device = torch.device("cpu")
    print("Running on the CPU")
    
# Writer will output to ./runs/ directory by default
writer = SummaryWriter("Res")
# move tensorboard files to C:\Users\FromM\Runs. Then run commond in cmd: tensorboard --logdir=C:\Users\FromM\Runs

start_time = time.time()
# fix random seed
np.random.seed(0)
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# =============================================================================
# load the data
# =============================================================================
data = scipy.io.loadmat(os.path.dirname(os.path.dirname(os.getcwd())) + '\\Lorenz_20ICs_15sec_dt0001_InitLessArbitrary.mat')
X = data['X'] # first dim: time steps. second dim: x y z. third dim: ICs
t = data['t']

# backup clean data
X_ref = copy.deepcopy(X)

# add noise
NoiseLvl = 0
for it_IC in range(X.shape[2]):
    X[:, :, it_IC] += NoiseLvl*np.std(X[:, :, it_IC], axis = 0, keepdims = True)*np.random.randn(X.shape[0], X.shape[1])

# normalize x, y, z to prevent gradient explosion
norm_factor = 10
X = X/norm_factor
X_ref = X_ref/norm_factor

# collect some meas
# equally spaced downsampling
meas_step = 10
ind_meas = np.arange(start = 0, stop = t.shape[0], step = meas_step) 

X_meas = torch.from_numpy(X[ind_meas, :, :]).to(device).type(torch.float32)
t_meas = torch.from_numpy(t[ind_meas, :]).to(device).type(torch.float32)

# training data
X_tr = X_meas[:, :, :10]
X_ref_tr = X_ref[:, :, :10]

# val data
X_val = X_meas[:, :, 10:15]
X_ref_val = X_ref[:, :, 10:15]

# test data
X_test = X_meas[:, :, 15:20]
X_ref_test = X_ref[:, :, 15:20]

dt = t[1, 0] - t[0, 0] # timestep size of simulated data


# =============================================================================
# training
# =============================================================================
# Type 2: SNN
D_in_r = 4
H = [2, 2] # hidden nodes for special ops
No_BiOp = [1, 1]
D_out_r = 1
SNN_x = SymbolicNet(D_in_r, H, D_out_r, No_BiOp).to(device)
SNN_y = SymbolicNet(D_in_r, H, D_out_r, No_BiOp).to(device)
SNN_z = SymbolicNet(D_in_r, H, D_out_r, No_BiOp).to(device)

# =============================================================================
# Pretrain
# =============================================================================
# load trained model
# checkpoint = torch.load('Pretrain.tar')
# SNN_x.load_state_dict(checkpoint['SNN_x_state_dict'])
# SNN_y.load_state_dict(checkpoint['SNN_y_state_dict'])
# SNN_z.load_state_dict(checkpoint['SNN_z_state_dict'])

Adam_epochs = 40000
sparse_coeff = 0
lr = 1e-3
SNN_x, SNN_y, SNN_z = AdamTrain(SNN_x, SNN_y, SNN_z, X_tr, X_val, dt, meas_step, lr, Adam_epochs, writer, 'Pre',
                                lr_decay = True)
# save model
torch.save({
            'SNN_x_state_dict': SNN_x.state_dict(),
            'SNN_y_state_dict': SNN_y.state_dict(),
            'SNN_z_state_dict': SNN_z.state_dict()
            }, 'Pretrain.tar')

# =============================================================================
# L1 train
# =============================================================================
Adam_epochs = 10000
sparse_coeff = 1e-7
lr = 1e-2
SNN_x, SNN_y, SNN_z = AdamTrain(SNN_x, SNN_y, SNN_z, X_tr, X_val, dt, meas_step, lr, Adam_epochs, writer, 'Sparse',
                                lr_decay = False, sparse_coeff = sparse_coeff)
# save model
torch.save({
            'SNN_x_state_dict': SNN_x.state_dict(),
            'SNN_y_state_dict': SNN_y.state_dict(),
            'SNN_z_state_dict': SNN_z.state_dict()
            }, 'Sparse.tar')

# =============================================================================
# results
# =============================================================================
elapsed = time.time() - start_time  
print('Training time: %.4f \n' % (elapsed))
writer.add_text('Time', 'Training time:' + str(elapsed))

# load trained model
# checkpoint = torch.load('Sparse.tar')
# SNN_x.load_state_dict(checkpoint['SNN_x_state_dict'])
# SNN_y.load_state_dict(checkpoint['SNN_y_state_dict'])
# SNN_z.load_state_dict(checkpoint['SNN_z_state_dict'])

# forecast training data
# NewMeasInterval = meas + windwon.
NewMeasInterval = 10
X_tr_all, error_tr_all = Predict_SNN_HiddenTS(SNN_x, SNN_y, SNN_z, X_tr, X_ref_tr, dt, meas_step, NewMeasInterval = NewMeasInterval)
error_tr_mean = np.mean(error_tr_all)
writer.add_text('Average Train Error(%)', 'Train Error:' + str(error_tr_mean))
print('Average Train Error(percentage): %.4f \n' % (error_tr_mean))

# forecast val data
X_val_all, error_val_all = Predict_SNN_HiddenTS(SNN_x, SNN_y, SNN_z, X_val, X_ref_val, dt, meas_step, NewMeasInterval = NewMeasInterval)
error_val_mean = np.mean(error_val_all)
writer.add_text('Average Val Error(%)', 'Val Error:' + str(error_val_mean))
print('Average Val Error(percentage): %.4f \n' % (error_val_mean))

# forecast test data
X_test_all, error_test_all = Predict_SNN_HiddenTS(SNN_x, SNN_y, SNN_z, X_test, X_ref_test, dt, meas_step, NewMeasInterval = NewMeasInterval)
error_test_mean = np.mean(error_test_all)
writer.add_text('Average Test Error(%)', 'Test Error:' + str(error_test_mean))
print('Average Test Error(percentage): %.4f \n' % (error_test_mean))

scipy.io.savemat('PredSol.mat', {'X_tr_pred': X_tr_all, 'X_tr': X_ref_tr, 'X_tr_n': X_tr.detach().cpu().numpy(),
                                 'X_val_pred': X_val_all, 'X_val': X_ref_val, 'X_val_n': X_val.detach().cpu().numpy(),
                                 'X_test_pred': X_test_all, 'X_test': X_ref_test, 'X_test_n': X_test.detach().cpu().numpy(),
                                 't': t, 'NewMeasInterval': NewMeasInterval, 'meas_step':meas_step})


## Predict w/ statistically different test datasets
# load trained model
# checkpoint = torch.load('Sparse.tar')
# SNN_x.load_state_dict(checkpoint['SNN_x_state_dict'])
# SNN_y.load_state_dict(checkpoint['SNN_y_state_dict'])
# SNN_z.load_state_dict(checkpoint['SNN_z_state_dict'])

# load test data
# data = scipy.io.loadmat(os.path.dirname(os.getcwd()) + '\\Lorenz_5ICs_15sec_dt0001_InitWild.mat')
data = scipy.io.loadmat(os.path.dirname(os.path.dirname(os.getcwd())) + '\\Lorenz_5ICs_15sec_dt0001_InitWild.mat')
X = data['X'] # first dim: time steps. second dim: x y z. third dim: ICs
t = data['t']

# backup clean data
X_ref = copy.deepcopy(X)

# add noise
NoiseLvl = 0
for it_IC in range(X.shape[2]):
    X[:, :, it_IC] += NoiseLvl*np.std(X[:, :, it_IC], axis = 0, keepdims = True)*np.random.randn(X.shape[0], X.shape[1])

# normalize x, y, z to prevent gradient explosion
norm_factor = 10
X = X/norm_factor
X_ref = X_ref/norm_factor

# collect some meas
# equally spaced downsampling
meas_step = 10
ind_meas = np.arange(start = 0, stop = t.shape[0], step = meas_step) 

X_meas = torch.from_numpy(X[ind_meas, :, :]).to(device).type(torch.float32)
t_meas = torch.from_numpy(t[ind_meas, :]).to(device).type(torch.float32)

# test data
X_test = X_meas
X_ref_test = X_ref

dt = t[1, 0] - t[0, 0] # timestep size of simulated data

# # forecast test data
NewMeasInterval = 10
X_test_all, error_test_all = Predict_SNN_HiddenTS(SNN_x, SNN_y, SNN_z, X_test, X_ref_test, dt, meas_step, NewMeasInterval = NewMeasInterval)
error_test_mean = np.mean(error_test_all)
writer.add_text('Average Test Error(%)', 'Test Error:' + str(error_test_mean))
print('Average Test Error(percentage): %.4f \n' % (error_test_mean))

scipy.io.savemat('PredSol_DifferentTest.mat', {
                                  'X_test_pred': X_test_all, 'X_test': X_ref_test, 'X_test_n': X_test.detach().cpu().numpy(),
                                  't': t, 'NewMeasInterval': NewMeasInterval, 'meas_step':meas_step})


writer.flush()
writer.close()


