import numpy as np
import torch.nn as nn
import torchpwl
import torch


def PhaseComDoppler_AF(inp,fr_size):
    final=torch.fft.ifftn(inp,fr_size[0] ,dim=2)
    [B,N,M]=final.shape
    pref=final[:,0,:]
    out=torch.zeros(final.shape).type(torch.complex64)
    out[:,0,:]=final[:,0,:]
    for k in range(N-1):
        delta=pref.conj()*final[:,k+1,:]
        phase=torch.sum(delta,dim=-1,keepdim=True)/torch.abs(torch.sum(delta,dim=-1,keepdim=True))
        out[:,k+1,:]=final[:,k+1,:]*phase.conj()
        if k<7:
            pref=pref+out[:,k+1,:]
        else:
            pref=pref+out[:,k+1,:]-out[:,k-7,:]
    out=torch.fft.fftn(out,fr_size[1],dim=1)
    # plt.figure()
    # plt.imshow(abs(out[0].detach().cpu().numpy()))
    # plt.show()
    return out

class ADMMCSNetLayer(nn.Module):
    def __init__(
        self,
        mask,
        args,
        in_channels: int = 1,
        out_channels: int = 1,
        kernel_size: int = 5

    ):
        """
        Args:

        """
        super(ADMMCSNetLayer, self).__init__()

        self.rho = nn.Parameter(torch.tensor([0.1]), requires_grad=True)
        self.gamma = nn.Parameter(torch.tensor([1.0]), requires_grad=True)
        self.mask = mask
        self.re_org_layer = ReconstructionOriginalLayer(self.rho, self.mask,args.fr_size)
        self.nonlinear_layer_ori = NonlinearLayer_ori()
        self.nonlinear_layer_mid = NonlinearLayer_mid()
        self.multiple_org_layer = MultipleOriginalLayer(self.gamma)
        self.re_update_layer = ReconstructionUpdateLayer(self.rho, self.mask,args.fr_size)
        self.add_layer = AdditionalLayer()
        self.multiple_update_layer = MultipleUpdateLayer(self.gamma)
        self.re_final_layer = ReconstructionFinalLayer(self.rho, self.mask,args.fr_size)
        layers = []

        layers.append(self.re_org_layer)
        layers.append(self.nonlinear_layer_ori)
        layers.append(self.multiple_org_layer)

        for i in range(8):
            layers.append(self.re_update_layer)
            layers.append(self.add_layer)
            layers.append(self.nonlinear_layer_mid)
            layers.append(self.multiple_update_layer)

        layers.append(self.re_update_layer)
        layers.append(self.add_layer)
        layers.append(self.nonlinear_layer_mid)
        layers.append(self.multiple_update_layer)

        layers.append(self.re_final_layer)

        self.cs_net = nn.Sequential(*layers)


    def forward(self, inp):
        x=inp[:,0,:,:]+1j*inp[:,1,:,:]
        y = torch.mul(x, self.mask)
        x = self.cs_net(y)
        return x


# reconstruction original layers
class ReconstructionOriginalLayer(nn.Module):
    def __init__(self, rho, mask,fr_size):
        super(ReconstructionOriginalLayer,self).__init__()
        self.rho = rho
        self.mask = 1
        self.fr_size=(fr_size[0],fr_size[1])

    def forward(self, x):
        mask = self.mask
        denom = torch.add(mask, self.rho)
        a = 1e-6
        value = torch.full(denom.size(), a)
        denom = torch.where(denom == 0, value, denom)
        orig_output1 = torch.div(1, denom)

        orig_output2 = torch.mul(x, orig_output1)
        orig_output3=PhaseComDoppler_AF(orig_output2.transpose(-1,-2),self.fr_size).transpose(-1,-2)
        # plt.figure()
        # plt.imshow(torch.abs(orig_output3[0]).detach().cpu().numpy())
        # plt.show()
        # orig_output3_tmp = torch.fft.ifftn(orig_output2,self.fr_size[0],dim=-2)
        # orig_output3=torch.fft.fftn(orig_output3_tmp,self.fr_size[1],dim=-1)
        # define data dict
        cs_data = dict()
        cs_data['input'] = x
        cs_data['nnline_input'] = orig_output3
        return cs_data


# reconstruction middle layers
class ReconstructionUpdateLayer(nn.Module):
    def __init__(self, rho, mask,fr_size):
        super(ReconstructionUpdateLayer,self).__init__()
        self.rho = rho
        self.mask = 1
        self.fr_size = (fr_size[0], fr_size[1])
        self.data_size=(fr_size[0]//2, fr_size[1]//2)

    def forward(self, x):
        minus_output = x['nnline_output']
        multiple_output = x['multi_output']
        input = x['input']
        rec_x=x['nnline_input']
        # spc=torch.fft.fft2(rec_x,self.data_size)
        spc_tmp=torch.fft.ifftn(rec_x,self.data_size[1],dim=-1)
        spc=torch.fft.fftn(spc_tmp,self.data_size[0],dim=-2)
        col_num=spc.shape[-1]
        phi=torch.zeros(spc.shape[0],col_num)
        for iter in range(col_num):
            tmp=torch.matmul(spc[:,:,iter].unsqueeze(-1).transpose(-1,-2).conj(),input[:,:,iter].unsqueeze(-1)).squeeze(-1)
            phi[:,iter]=-torch.atan2(tmp.real,tmp.imag).squeeze(-1)
        E=torch.diag_embed(torch.exp(1j*phi)).to(input.device)
        mask = self.mask
        minus=torch.sub(minus_output, multiple_output)
        rec_=torch.matmul(input,E.conj())
        rec_ = torch.fft.ifftn(rec_,self.fr_size[0],dim=-2)
        rec_=torch.fft.fftn(rec_,self.fr_size[1],dim=-1)
        number=minus-rec_
        denom = torch.add(mask, self.rho)
        a = 1e-6
        value = torch.full(denom.size(), a)
        denom = torch.where(denom == 0, value, denom)
        orig_output1 = torch.div(1, denom)
        orig_output2 = torch.mul(number, orig_output1)
        x['re_mid_output'] = minus-orig_output2
        return x


# reconstruction middle layers
class ReconstructionFinalLayer(nn.Module):
    def __init__(self, rho, mask,fr_size):
        super(ReconstructionFinalLayer, self).__init__()
        self.rho = rho
        self.mask = 1
        self.fr_size=(fr_size[0],fr_size[1])
        self.data_size=(fr_size[0]//2, fr_size[1]//2)

    def forward(self, x):
        minus_output = x['nnline_output']
        multiple_output = x['multi_output']
        input = x['input']
        rec_x=x['nnline_input']
        # spc=torch.fft.fft2(rec_x,self.data_size)
        spc_tmp = torch.fft.ifftn(rec_x, self.data_size[1], dim=-1)
        spc = torch.fft.fftn(spc_tmp, self.data_size[0], dim=-2)
        col_num=spc.shape[-1]
        phi=torch.zeros(spc.shape[0],col_num)
        for iter in range(col_num):
            tmp=torch.matmul(spc[:,:,iter].unsqueeze(-1).transpose(-1,-2).conj(),input[:,:,iter].unsqueeze(-1)).squeeze(-1)
            phi[:,iter]=-torch.atan2(tmp.real,tmp.imag).squeeze(-1)
        E=torch.diag_embed(torch.exp(1j*phi)).to(input.device)
        mask = self.mask
        minus=torch.sub(minus_output, multiple_output)
        rec_ = torch.matmul(input, E.conj())
        rec_ = torch.fft.ifftn(rec_, self.fr_size[0], dim=-2)
        rec_ = torch.fft.fftn(rec_, self.fr_size[1], dim=-1)
        number = minus - rec_
        denom = torch.add(mask, self.rho)
        a = 1e-6
        value = torch.full(denom.size(), a)
        denom = torch.where(denom == 0, value, denom)
        orig_output1 = torch.div(1, denom)
        orig_output2 = torch.mul(number, orig_output1)
        x['re_final_output'] = minus-orig_output2
        return x['re_final_output']


# multiple original layer
class MultipleOriginalLayer(nn.Module):
    def __init__(self,gamma):
        super(MultipleOriginalLayer,self).__init__()
        self.gamma = gamma

    def forward(self,x):
        org_output = x['nnline_input']
        minus_output = x['nnline_output']
        output= torch.mul(self.gamma,torch.sub(org_output, minus_output))
        x['multi_output'] = output
        return x


# multiple middle layer
class MultipleUpdateLayer(nn.Module):
    def __init__(self,gamma):
        super(MultipleUpdateLayer,self).__init__()
        self.gamma = gamma

    def forward(self, x):
        multiple_output = x['multi_output']
        re_mid_output = x['re_mid_output']
        minus_output = x['nnline_output']
        output= torch.add(multiple_output,torch.mul(self.gamma,torch.sub(re_mid_output , minus_output)))
        x['multi_output'] = output
        return x




# nonlinear layer
class NonlinearLayer_ori(nn.Module):
    def __init__(self):
        super(NonlinearLayer_ori,self).__init__()
        self.pwl = torchpwl.PWL(num_channels=1, num_breakpoints=101)

    def forward(self, x):
        nnline_input = x['nnline_input'].unsqueeze(1)
        y_real = self.pwl(nnline_input.real)
        y_imag = self.pwl(nnline_input.imag)
        output = torch.complex(y_real, y_imag).squeeze(1)
        x['nnline_output'] = output
        return x

class NonlinearLayer_mid(nn.Module):
    def __init__(self):
        super(NonlinearLayer_mid,self).__init__()
        self.pwl = torchpwl.PWL(num_channels=1, num_breakpoints=101)

    def forward(self, x):
        nnline_input = x['add_output'].unsqueeze(1)
        y_real = self.pwl(nnline_input.real)
        y_imag = self.pwl(nnline_input.imag)
        output = torch.complex(y_real, y_imag).squeeze(1)
        x['nonlinear_output'] = output
        return x


# addtional layer
class AdditionalLayer(nn.Module):
    def __init__(self):
        super(AdditionalLayer,self).__init__()

    def forward(self, x):
        mid_output = x['re_mid_output']
        multi_output = x['multi_output']
        output= torch.add(mid_output,multi_output)
        x['add_output'] = output
        return x
