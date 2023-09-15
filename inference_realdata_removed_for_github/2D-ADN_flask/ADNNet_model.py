import torch
import util_2D_3 as util
import numpy as np
import h5py
import os
import time
import argparse
from CSNet_Layers import ADMMCSNetLayer
from CSNet_Layers2 import ADMMCSNetLayer2
class Trainer:

    def __init__(self,data_path):
        parser = argparse.ArgumentParser(description=' test ')
        parser.add_argument('--signal_dim', default=[256, 64], type=list)
        parser.add_argument('--fr_size', default=[512, 128], type=list)
        parser.add_argument('--outf', type=str, default='model', help='path of log files')
        args = parser.parse_args()

        self.model = ADMMCSNetLayer(1, args)
        self.model.load_state_dict(torch.load(os.path.join(args.outf, 'model60.pth'),map_location=torch.device('cpu')))
        self.model.cpu()
        self.model.eval()
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
        signal_50dB=np.zeros([1,2,N,M]).astype(np.float32)
        signal_50dB[0,0]=(real_data.astype(np.float32))
        signal_50dB[0,1] = (imag_data.astype(np.float32))

        noisy_signal = torch.tensor(signal_50dB[idx][None]).float()

        with torch.no_grad():

            pre_50dB=self.model(noisy_signal).abs().transpose(-1,-2)



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





