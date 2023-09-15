import torch
import util_2D_3 as util
import numpy as np
import h5py
import os
import time


class Trainer:

    def __init__(self,data_path):
        skip_path = os.path.join('model', 'epoch_270.pth')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.skip_module, _, _, _, _ = util.load(skip_path, 'pre',device)
        self.skip_module.cpu()
        self.skip_module.eval()
        self.data_path=data_path


    def inference(self):
        path = os.path.join(self.data_path, 'matlab_real.h5')
        f = h5py.File(path, 'r')
        real_data = f['matlab_real'][:]
        f.close()
        path = os.path.join(self.data_path, 'matlab_imag.h5')
        f = h5py.File(path, 'r')
        imag_data = f['matlab_imag'][:]
        f.close()
        #path = os.path.join(self.data_path, 'bz.h5')
        #f = h5py.File(path, 'r')
        #bz = f['bz'][:]
        #f.close()

        M=64
        N=256
        idx=0

        tt = time.time()
        signal_50dB=np.zeros([1,2,M,N]).astype(np.float32)
        signal_50dB[0,0]=(real_data.astype(np.float32)).T
        signal_50dB[0,1] = (imag_data.astype(np.float32)).T

        noisy_signal = torch.tensor(signal_50dB[idx][None]).float()

        with torch.no_grad():

            pre_50dB=self.skip_module(noisy_signal)



        pre_50dB = pre_50dB.numpy()
        pre_50dB=pre_50dB.squeeze(-3)
        cost = time.time() - tt

        path = os.path.join(self.data_path, 'python_data.h5')
        f = h5py.File(path, 'w') 
        f['python_data'] =pre_50dB
        f.close()

        path = os.path.join(self.data_path, 'cost.h5')
        f = h5py.File(path, 'w')
        f['cost'] =cost
        f.close()  





