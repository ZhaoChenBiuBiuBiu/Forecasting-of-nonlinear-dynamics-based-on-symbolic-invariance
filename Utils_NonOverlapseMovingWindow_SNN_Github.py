# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 11:35:42 2020

@author: FromM
"""
import torch
import numpy as np
from tqdm import tqdm

# =============================================================================
# Functions
# =============================================================================
def ResMdl_SNN(X_t, SNN_x, SNN_y, SNN_z, dt):    
    # explicit rk4
    k1 = SNN_deri(X_t, SNN_x, SNN_y, SNN_z)
        
    y_k2 = X_t + dt/2*k1
    k2 = SNN_deri(y_k2, SNN_x, SNN_y, SNN_z)
    
    y_k3 = X_t + dt/2*k2
    k3 = SNN_deri(y_k3, SNN_x, SNN_y, SNN_z)
    
    y_k4 = X_t + dt*k3
    k4 = SNN_deri(y_k4, SNN_x, SNN_y, SNN_z)
    
    X_t1 = X_t + 1/6*dt*(k1 + 2*k2 + 2*k3 + k4)
    
    return X_t1

def ResMdl_SNN_backward(X_t, SNN_x, SNN_y, SNN_z, dt):    
    # explicit rk4
    k1 = SNN_deri(X_t, SNN_x, SNN_y, SNN_z)
        
    y_k2 = X_t - dt/2*k1
    k2 = SNN_deri(y_k2, SNN_x, SNN_y, SNN_z)
    
    y_k3 = X_t - dt/2*k2
    k3 = SNN_deri(y_k3, SNN_x, SNN_y, SNN_z)
    
    y_k4 = X_t - dt*k3
    k4 = SNN_deri(y_k4, SNN_x, SNN_y, SNN_z)
    
    X_t1 = X_t - 1/6*dt*(k1 + 2*k2 + 2*k3 + k4)
    
    return X_t1


def AdamTrain(SNN_x, SNN_y, SNN_z, X_tr, X_val, dt, meas_step, lr, Adam_epochs, writer, Phase, lr_decay = False,
              sparse_coeff = 0):
    optimizer = torch.optim.Adam(SNN_x.linear_weights_all + SNN_y.linear_weights_all + SNN_z.linear_weights_all, lr)
    if lr_decay:
        # scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, lr, total_steps=Adam_epochs,
        #                                             final_div_factor=1e1)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor = 0.75)
        
    
    # format sparse measurements into non-overlapse moving window
    X_input_tr, X_output_forward_tr, X_output_backward_tr = FormatInputOutput(X_tr)
    X_input_val, X_output_forward_val, X_output_backward_val = FormatInputOutput(X_val)
    for it_Adam in tqdm(range(Adam_epochs)):   
        loss_X_tr = AdamTrain_Forward(X_input_tr, meas_step, SNN_x, SNN_y, SNN_z, dt, X_output_forward_tr, X_output_backward_tr)
        loss_X_val = AdamTrain_Forward(X_input_val, meas_step, SNN_x, SNN_y, SNN_z, dt, X_output_forward_val, X_output_backward_val)
                
        # L1 penalty
        x_para = 0
        y_para = 0
        z_para = 0
        for para in SNN_x.linear_weights_all:
            x_para += torch.sum(torch.abs(para))
        for para in SNN_y.linear_weights_all:
            y_para += torch.sum(torch.abs(para))
        for para in SNN_z.linear_weights_all:
            z_para += torch.sum(torch.abs(para))

        
        if sparse_coeff == 0:
            loss = loss_X_tr
        else:
            loss = loss_X_tr + sparse_coeff*(x_para + y_para + z_para)
            
        
        if it_Adam % 10 == 0:
            writer.add_scalar('loss_X_tr_' + Phase, loss_X_tr.item(), it_Adam)
            writer.add_scalar('loss_X_val_' + Phase, loss_X_val.item(), it_Adam)
            writer.add_scalar('x_para_' + Phase, x_para, it_Adam)
            writer.add_scalar('y_para_' + Phase, y_para, it_Adam)
            writer.add_scalar('z_para_' + Phase, z_para, it_Adam)
        optimizer.zero_grad()
    
        loss.backward()       
    
        optimizer.step()
    
        if lr_decay:
            scheduler.step(loss_X_val) 

    return SNN_x, SNN_y, SNN_z

def AdamTrain_Forward(X_input_tr, meas_step, SNN_x, SNN_y, SNN_z, dt, X_output_forward, X_output_backward):
    loss_fn = torch.nn.MSELoss()

    ## type 4: concatenate data of all moving windows for one epoch, while fix the recurrency to PredWindowSize
    X_pred_forward_list = [X_input_tr]
    X_pred_backward_list = [X_input_tr]
    for it_measstep in range(meas_step): # loop to propagate on unmeasured pts
        X_pred_forward_list.append(ResMdl_SNN(X_pred_forward_list[it_measstep], SNN_x, SNN_y, SNN_z, dt)) # dim 1: window*IC. dim 2: x, y, z. list dim : pred window size
        X_pred_backward_list.append(ResMdl_SNN_backward(X_pred_backward_list[it_measstep], SNN_x, SNN_y, SNN_z, dt)) # dim 1: window*IC. dim 2: x, y, z. list dim: pred window size
    # the last prediction pt corresponds to a meas pt. compare them.
    X_pred_forward = X_pred_forward_list[-1] # dim 1: window*IC. dim 2: x, y, z.
    X_pred_backward = X_pred_backward_list[-1] # dim 1: window*IC. dim 2: x, y, z.

    loss_x = loss_fn(X_pred_forward[:, :1], X_output_forward[:, :1]) + loss_fn(X_pred_backward[:, :1], X_output_backward[:, :1])
    loss_y = loss_fn(X_pred_forward[:, 1:2], X_output_forward[:, 1:2]) + loss_fn(X_pred_backward[:, 1:2], X_output_backward[:, 1:2])
    loss_z = loss_fn(X_pred_forward[:, 2:], X_output_forward[:, 2:]) + loss_fn(X_pred_backward[:, 2:], X_output_backward[:, 2:])

    return loss_x + loss_y + loss_z

def FormatInputOutput(X_tr):
    # format sparse measurements into non-overlapse moving window
    start_ind = 1
    InputWindows = X_tr[start_ind:-1, :, :] # window no.*sys dim*IC no. Get rid of the first and the last timestep since we want bi-direction prediction
    OutputWindows_forward_list = []
    OutputWindows_backward_list = []
    for it_N in range(InputWindows.shape[0]):        
        OutputWindows_forward_list.append(X_tr[start_ind + it_N + 1:start_ind + it_N + 2, :, :])
        OutputWindows_backward_list.append(X_tr[start_ind + it_N - 1:start_ind + it_N, :, :])
    OutputWindows_forward = torch.cat(OutputWindows_forward_list, axis = 0) # dim 1: window no. * timestep. dim 2: sys dim. dim 3: IC
    OutputWindows_backward = torch.cat(OutputWindows_backward_list, axis = 0) # dim 1: window no. * timestep. dim 2: sys dim. dim 3: IC
    X_input = torch.reshape(InputWindows.permute(2, 0, 1), (-1, X_tr.shape[1])) # dim 1: window*IC. dim 2: x, y, z.
    X_output_forward = torch.reshape(OutputWindows_forward.permute(2, 0, 1), (-1, X_tr.shape[1])) # dim 1: window*IC. dim 2: x, y, z.
    X_output_backward = torch.reshape(OutputWindows_backward.permute(2, 0, 1), (-1, X_tr.shape[1])) # dim 1: window*IC. dim 2: x, y, z.
    return X_input, X_output_forward, X_output_backward

def SNN_deri(X_t, SNN_x, SNN_y, SNN_z):
    SNNInput = torch.cat([torch.ones_like(X_t[:, :1]), X_t,
                          ], axis = 1)
    dx_t = SNN_x(SNNInput)
    dy_t = SNN_y(SNNInput)
    dz_t = SNN_z(SNNInput)
    
    dX_t = torch.cat([dx_t, dy_t, dz_t], axis = 1)
    return dX_t
    

def Predict_SNN_HiddenTS(SNN_x, SNN_y, SNN_z, X, X_ref, dt, meas_step, NewMeasInterval):
    # NewMeasInterval = starting meas + window.
    X_all_IC = [] 
    error_all_IC = []    
    
    interval_ind_array = np.arange(start = 0, stop = X.shape[0], step = NewMeasInterval)
    for it_IC in range(X.shape[2]):
        X_pred_IC = []
        for it_Interval in range(interval_ind_array.shape[0]):
            interval_ind = interval_ind_array[it_Interval]
            X_pred_interval = [X[interval_ind:interval_ind + 1, :, it_IC]] # this windows
            if it_Interval == interval_ind_array.shape[0] - 1:
                window_length = X.shape[0] - interval_ind 
            else:
                window_length = interval_ind_array[it_Interval + 1] - interval_ind
            for it in range(window_length*meas_step - 1): # exclude next meas starting pt
                X_pred_interval.append(ResMdl_SNN(X_pred_interval[it], SNN_x, SNN_y, SNN_z, dt))                    
            X_pred_IC.append(torch.cat(X_pred_interval, 0))
        
        X_pred_thisIC = torch.cat(X_pred_IC, 0).detach().cpu().numpy()
        X_all_IC.append(X_pred_thisIC)

        X_ref_thisIC = X_ref[:, :, it_IC]
        error = np.linalg.norm(X_pred_thisIC - X_ref_thisIC)/np.linalg.norm(X_ref_thisIC)*100
        error_all_IC.append(error)
    
    X_pred_all = np.stack(X_all_IC, axis = -1)
    error_all = np.stack(error_all_IC)
 
    return X_pred_all, error_all
def weights_init(m):
    torch.nn.init.normal_(m.weight, std = 1e-2)

# =============================================================================
# Classes
# =============================================================================
class SymbolicNet(torch.nn.Module):
    def __init__(self, D_in_r, H, D_out_r, No_BiOp):
        super(SymbolicNet, self).__init__()
        self.linear1 = torch.nn.Linear(D_in_r, H[0], bias = False)
        self.linear2 = torch.nn.Linear(H[0] - No_BiOp[0] + D_in_r, H[1], bias = False) 
        self.linear3 = torch.nn.Linear(H[1] - No_BiOp[1] + D_in_r + No_BiOp[0], D_out_r, bias = False)

        self.linear_weights_all = [self.linear1.weight, self.linear2.weight, self.linear3.weight]
        self.No_BiOp = No_BiOp
        self.H = H
        self.D_in_r = D_in_r

        self.scale_fac = 1
        
        weights_init(self.linear1)
        weights_init(self.linear2)
        weights_init(self.linear3)

    def forward(self, X):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """        
        
        h1 = self.scale_fac*self.linear1(X)
        h1_a = torch.cat((X,
                            h1[:, :1]*h1[:, 1:2]), 1)
        
        h2 = self.scale_fac*self.linear2(h1_a)
        
        # inherit ids and muls from h1_a
        h2_a = torch.cat((h1_a,
                            h2[:, :1]*h2[:, 1:2]), 1)

        h3 = self.scale_fac*self.linear3(h2_a)
        
        return h3
