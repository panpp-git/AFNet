import torch.nn as nn
import torch
from complexLayers import ComplexLinear,ComplexReLU,ComplexConv1d,ComplexConvTranspose1d,ComplexConv2d,ComplexConvTranspose2d,ComplexBatchNorm2d
from complexLayer2 import ComplexBatchNorm2d_2,ComplexReLU2,ComplexConv2d2,ComplexReLU3




def set_pre_module(args):
    net = None
    if args.fr_module_type == 'fr':
        # net = FrequencyRepresentationModule_2D_v0523(params=args)
        net = FrequencyRepresentationModule_2D_v6(params=args)
    else:
        raise NotImplementedError('Frequency representation module type not implemented')
    if args.use_cuda:
        net.cuda()
    return net
import math
class REDNet30(nn.Module):
    def __init__(self, num_layers=15, num_features=8):
        super(REDNet30, self).__init__()
        self.num_layers = num_layers

        conv_layers = []
        deconv_layers = []

        conv_layers.append(nn.Sequential(nn.Conv2d(num_features, num_features*2, kernel_size=3, stride=2, padding=1),
                                         nn.ReLU(inplace=True)))
        for i in range(num_layers - 1):
            conv_layers.append(nn.Sequential(nn.Conv2d(num_features*2, num_features*2, kernel_size=3, padding=1),
                                             nn.ReLU(inplace=True)))

        for i in range(num_layers - 1):
            deconv_layers.append(nn.Sequential(nn.ConvTranspose2d(num_features*2, num_features*2, kernel_size=3, padding=1),
                                               nn.ReLU(inplace=True)))
        deconv_layers.append(nn.ConvTranspose2d(num_features*2, num_features, kernel_size=3, stride=2, padding=1, output_padding=1))

        self.conv_layers = nn.Sequential(*conv_layers)
        self.deconv_layers = nn.Sequential(*deconv_layers)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x

        conv_feats = []
        for i in range(self.num_layers):
            x = self.conv_layers[i](x)
            if (i + 1) % 2 == 0 and len(conv_feats) < math.ceil(self.num_layers / 2) - 1:
                conv_feats.append(x)

        conv_feats_idx = 0
        for i in range(self.num_layers):
            x = self.deconv_layers[i](x)
            if (i + 1 + self.num_layers) % 2 == 0 and conv_feats_idx < len(conv_feats):
                conv_feat = conv_feats[-(conv_feats_idx + 1)]
                conv_feats_idx += 1
                x = x + conv_feat
                x = self.relu(x)

        x += residual
        x = self.relu(x)

        return x
class FrequencyRepresentationModule_2D_v6(nn.Module):
    def __init__(self, params=None):
        super().__init__()
        self.A_num=2
        self.n_layer=params.fr_n_layers
        n_filters=self.A_num
        self.dim0 = params.fr_inner_dim_0
        self.dim1 = params.fr_inner_dim_1
        kernel_size=3
        self.input_layer =ComplexConv2d2(1, params.fr_inner_dim_1*self.A_num, kernel_size=(1,params.signal_dim_1), padding=0, bias=False)
        self.input_layer2=ComplexConv2d2(self.A_num, params.fr_inner_dim_0*self.A_num, kernel_size=(1,params.signal_dim_0), padding=0, bias=False)
        self.complex_conv=ComplexConv2d2(self.A_num, self.A_num, kernel_size=(params.signal_dim_1,1), bias=False)
        self.input_layer3 = ComplexConv2d2(self.A_num, params.fr_inner_dim_0 * self.A_num,
                                          kernel_size=(1, params.signal_dim_0), padding=0, bias=False)

        complex_mod1=[]
        ac1=[]
        bn1=[]
        for i in range(params.fr_n_layers//2):
            complex_mod1+=[ComplexConv2d2(n_filters,n_filters*params.signal_dim_0,kernel_size=(params.signal_dim_0,1),padding=0,bias=False)]
            bn1+=[ComplexBatchNorm2d_2(n_filters)]
            ac1 += [ComplexReLU2()]
        self.complex_mod1=nn.Sequential(*complex_mod1)
        self.ac1 = nn.Sequential(*ac1)
        self.bn1=nn.Sequential(*bn1)

        complex_mod2=[]
        ac2=[]
        bn2=[]
        for i in range(params.fr_n_layers//2):
            complex_mod2+=[ComplexConv2d2(n_filters,n_filters*params.signal_dim_0,kernel_size=(params.signal_dim_0,1),padding=0,bias=False)]
            bn2+=[ComplexBatchNorm2d_2(n_filters)]
            ac2 += [ComplexReLU2()]
        self.complex_mod2=nn.Sequential(*complex_mod2)
        self.ac2 = nn.Sequential(*ac2)
        self.bn2=nn.Sequential(*bn2)

        mod2=[]
        for i in range(params.fr_n_layers):
            tmp2 = []
            tmp2 += [
                nn.Conv2d(n_filters, n_filters, kernel_size=kernel_size, padding=kernel_size//2,
                                bias=False,padding_mode='circular'),
                nn.BatchNorm2d(n_filters),
                nn.ReLU(),
                nn.Conv2d(n_filters, n_filters, kernel_size=kernel_size, padding=kernel_size//2,
                              bias=False,padding_mode='circular'),
                nn.BatchNorm2d(n_filters),
            ]
            mod2+= [nn.Sequential(*tmp2)]
        self.mod2=nn.Sequential(*mod2)
        activate_layer2 = []
        for i in range(params.fr_n_layers):
            activate_layer2+=[nn.ReLU()]
        self.activate_layer2=nn.Sequential(*activate_layer2)
        self.out_layer = nn.ConvTranspose2d(n_filters, 1, kernel_size=params.fr_kernel_out, stride=params.fr_upsampling,
                                            padding=2, output_padding=1, bias=False)

    def forward(self, inp):

        bsz = inp.size(0)
        # inp1 = inp[:, 0].type(torch.complex64) + 1j * inp[:, 1].type(torch.complex64)
        # x=inp1.view(bsz,1,inp.size(2),-1)
        xreal,ximag = self.input_layer(inp)
        xreal=xreal.squeeze(-1).view(bsz,self.A_num,self.dim1,self.dim0)
        ximag = ximag.squeeze(-1).view(bsz, self.A_num, self.dim1, self.dim0)
        xcp_real=xreal.permute([0, 1, 3, 2])
        xcp_imag=ximag.permute([0,1,3,2])
        x1_real=xreal[:,:,:,0:self.dim0-1]
        x1_imag_conj=-1*ximag[:,:,:,0:self.dim0-1]
        x2_real=xreal[:,:,:,1:self.dim0]
        x2_imag=ximag[:,:,:,1:self.dim0]
        x_new_real=x1_real*x2_real-x1_imag_conj*x2_imag
        x_new_imag=x1_real*x2_imag+x1_imag_conj*x2_real
        x_new=torch.cat((x_new_real,x_new_imag),1)
        compx_real,compx_imag = self.complex_conv(x_new)
        compx_real=compx_real.squeeze(-2)
        compx_imag=compx_imag.squeeze(-2)
        angle_compx = torch.atan2(compx_imag, compx_real)
        angle_compx2 = torch.zeros([bsz, self.A_num, self.dim0]).to(inp.device)
        angle_compx2[:, :, 1:self.dim0] = angle_compx
        angle_compx = torch.cumsum(angle_compx2, dim=2).view(bsz * self.A_num * self.dim0, -1)
        angle_compx = angle_compx.repeat(1, self.dim1).view(bsz, self.A_num, self.dim0, -1)
        compx_real=torch.cos(angle_compx)
        compx_imag_conj=-1*torch.sin(angle_compx)
        x_real=xcp_real*compx_real-xcp_imag*compx_imag_conj
        x_imag=xcp_real*compx_imag_conj+xcp_imag*compx_real
        Gnk_real=x_real
        Gnk_imag=x_imag
        x_real=x_real.permute(0,1,3,2)
        x_imag = x_imag.permute(0, 1, 3, 2)
        x=torch.cat((x_real,x_imag),1)
        x_real,x_imag=self.input_layer2(x)
        x_real=x_real.squeeze(-1).view(bsz,-1)
        x_imag=x_imag.squeeze(-1).view(bsz,-1)
        xabs=torch.sqrt(torch.pow(x_real,2)+torch.pow(x_imag,2))+1e-6
        x_real=x_real/ torch.max(xabs, dim=1).values.repeat(xabs.size(1), 1).T
        x_imag=x_imag/ torch.max(xabs, dim=1).values.repeat(xabs.size(1), 1).T
        x_real=x_real.view(bsz, self.A_num, self.dim0, -1)
        x_imag = x_imag.view(bsz, self.A_num, self.dim0, -1)

        Iqk_real=x_real
        Iqk_Imag=x_imag
        # Iqk_real = Iqk_real.view(bsz, self.A_num, self.dim0, -1)
        # Iqk_Imag = Iqk_Imag.view(bsz, self.A_num, self.dim0, -1)

        for i in range(self.n_layer // 2):
            Iqk_abs=torch.sqrt(torch.pow(Iqk_real,2)+torch.pow(Iqk_Imag,2))
            EI_real=torch.log(Iqk_abs)*Iqk_real
            EI_imag=torch.log(Iqk_abs)*(-1)*Iqk_Imag
            EI=torch.cat((EI_real,EI_imag),1)
            xreal,ximag = self.complex_mod1[i](EI)
            xreal = xreal.squeeze(-2).view(bsz, self.A_num, self.dim0, -1)
            ximag = ximag.squeeze(-2).view(bsz, self.A_num, self.dim0, -1)
            x=torch.cat((xreal,ximag),1)
            x=self.bn1[i](x)
            x = self.ac1[i](x)
            xreal,ximag=torch.chunk(x,2,1)

            tmp_real=Gnk_real*xreal-Gnk_imag*ximag
            tmp_imag=Gnk_real*ximag+Gnk_imag*xreal
            wgt_real=torch.sum(tmp_real,dim=3).squeeze(-1)
            wgt_imag=torch.sum(tmp_imag,dim=3).squeeze(-1)
            angle_wgt = torch.atan2(wgt_imag, wgt_real)
            angle_wgt = angle_wgt.view(bsz * self.A_num * self.dim0, -1)
            angle_wgt = angle_wgt.repeat(1, self.dim1).view(bsz, self.A_num, self.dim0, -1)
            comp_wgt_real=torch.cos(angle_wgt)
            comp_wgt_imag=-1*torch.sin(angle_wgt)
            Gnk_real=Gnk_real*comp_wgt_real-Gnk_imag*comp_wgt_imag
            Gnk_imag=Gnk_real*comp_wgt_imag+Gnk_imag*comp_wgt_real
            Gnk=torch.cat((Gnk_real,Gnk_imag),1)
            Iqk_real,Iqk_imag = self.complex_mod2[i](Gnk)
            Iqk_real = Iqk_real.squeeze(-2).view(bsz, self.A_num, self.dim0, -1)
            Iqk_imag = Iqk_imag.squeeze(-2).view(bsz, self.A_num, self.dim0, -1)
            Iqk=torch.cat((Iqk_real,Iqk_imag),1)
            Iqk = self.bn2[i](Iqk)
            Iqk_real,Iqk_imag=torch.chunk(Iqk,2,1)
        x = Gnk.permute(0, 1, 3, 2)
        xreal,ximag = self.input_layer3(x)

        # xreal=xreal.squeeze(-1).view(bsz, self.A_num, self.dim0, -1)
        # ximag=ximag.squeeze(-1).view(bsz, self.A_num, self.dim0, -1)

        xreal = xreal.squeeze(-1).view(bsz, -1)
        ximag = ximag.squeeze(-1).view(bsz, -1)
        xabs=torch.sqrt(torch.pow(xreal,2)+torch.pow(ximag,2))
        xreal=xreal/torch.max(xabs, dim=1).values.repeat(xabs.size(1), 1).T
        ximag=ximag/torch.max(xabs, dim=1).values.repeat(xabs.size(1), 1).T
        xreal=xreal.view(bsz, self.A_num, self.dim0, -1)
        ximag = ximag.view(bsz, self.A_num, self.dim0, -1)


        x=torch.sqrt(torch.pow(xreal,2)+torch.pow(ximag,2))
        for i in range(self.n_layer):
            res_x = self.mod2[i](x)
            x = x + res_x
            x = self.activate_layer2[i](x)
        x = self.out_layer(x)
        x = x.squeeze(-3)
        return x


class FrequencyRepresentationModule_2D_v0417(nn.Module):
    def __init__(self, params=None):
        super().__init__()
        self.A_num=1
        self.n_layer=params.fr_n_layers
        n_filters=self.A_num
        self.dim0 = params.fr_inner_dim_0
        self.dim1 = params.fr_inner_dim_1
        kernel_size=3
        self.input_layer =ComplexConv2d2(1, params.fr_inner_dim_1*self.A_num, kernel_size=(1,params.signal_dim_1), padding=0, bias=False)
        self.complex_conv=ComplexConv2d2(self.A_num, self.A_num, kernel_size=(params.signal_dim_1,1), bias=False)
        self.input_layer3 = ComplexConv2d2(self.A_num, params.fr_inner_dim_0 * self.A_num,
                                          kernel_size=( params.signal_dim_0,1), padding=0, bias=False)

        complex_mod1=[]
        for i in range(params.fr_n_layers):
            complex_mod1+=[ComplexConv2d2(n_filters,n_filters*params.signal_dim_0,kernel_size=(params.signal_dim_0,1),padding=0,bias=False)]
        self.complex_mod1=nn.Sequential(*complex_mod1)


        mod2=[]
        for i in range(params.fr_n_layers):
            tmp2 = []
            tmp2 += [
                nn.Conv2d(n_filters, n_filters, kernel_size=kernel_size, padding=kernel_size//2,
                                bias=False,padding_mode='circular'),
                nn.BatchNorm2d(n_filters),
                nn.ReLU(),
                nn.Conv2d(n_filters, n_filters, kernel_size=kernel_size, padding=kernel_size//2,
                              bias=False,padding_mode='circular'),
                nn.BatchNorm2d(n_filters),
            ]
            mod2+= [nn.Sequential(*tmp2)]
        self.mod2=nn.Sequential(*mod2)
        activate_layer2 = []
        for i in range(params.fr_n_layers):
            activate_layer2+=[nn.ReLU()]
        self.activate_layer2=nn.Sequential(*activate_layer2)
        self.out_layer = nn.ConvTranspose2d(n_filters, 1, kernel_size=params.fr_kernel_out, stride=params.fr_upsampling,
                                            padding=2, output_padding=1, bias=False)

    def forward(self, inp):
        bsz = inp.size(0)
        xreal,ximag = self.input_layer(inp)
        xreal=xreal.squeeze(-1).view(bsz,self.A_num,self.dim1,self.dim0)
        ximag = ximag.squeeze(-1).view(bsz, self.A_num, self.dim1, self.dim0)
        xcp_real=xreal.permute([0, 1, 3, 2])
        xcp_imag=ximag.permute([0,1,3,2])
        # x1_real=xreal[:,:,:,0:self.dim0-1]
        # x1_imag_conj=-1*ximag[:,:,:,0:self.dim0-1]
        # x2_real=xreal[:,:,:,1:self.dim0]
        # x2_imag=ximag[:,:,:,1:self.dim0]
        # x_new_real=x1_real*x2_real-x1_imag_conj*x2_imag
        # x_new_imag=x1_real*x2_imag+x1_imag_conj*x2_real
        # x_new=torch.cat((x_new_real,x_new_imag),1)
        # compx_real,compx_imag = self.complex_conv(x_new)
        # compx_real=compx_real.squeeze(-2)
        # compx_imag=compx_imag.squeeze(-2)
        # angle_compx = torch.atan2(compx_imag, compx_real)
        # angle_compx2 = torch.zeros([bsz, self.A_num, self.dim0]).to(inp.device)
        # angle_compx2[:, :, 1:self.dim0] = angle_compx
        # angle_compx = torch.cumsum(angle_compx2, dim=2).view(bsz * self.A_num * self.dim0, -1)
        # angle_compx = angle_compx.repeat(1, self.dim1).view(bsz, self.A_num, self.dim0, -1)
        # compx_real=torch.cos(angle_compx)
        # compx_imag_conj=-1*torch.sin(angle_compx)
        # x_real=xcp_real*compx_real-xcp_imag*compx_imag_conj
        # x_imag=xcp_real*compx_imag_conj+xcp_imag*compx_real
        x_real=xcp_real
        x_imag=xcp_imag
        Gnk_real=x_real
        Gnk_imag=x_imag

        for i in range(self.n_layer):
            x = torch.cat((x_real, x_imag), 1)
            x_real, x_imag = self.complex_mod1[i](x)
            x_real = x_real.squeeze(-2)
            x_imag = x_imag.squeeze(-2)
            x_real = x_real.view(bsz, self.A_num, self.dim0, -1)
            x_imag = x_imag.view(bsz, self.A_num, self.dim0, -1)
            Iqk_real = x_real
            Iqk_Imag = x_imag

            Iqk_abs=torch.sqrt(torch.pow(Iqk_real,2)+torch.pow(Iqk_Imag,2))
            EI_real=(1+torch.log(Iqk_abs))*Iqk_real
            EI_imag=(1+torch.log(Iqk_abs))*(-1)*Iqk_Imag
            ur=self.complex_mod1[i].conv_r.weight[:,0,:,0].T
            ui=self.complex_mod1[i].conv_i.weight[:,0,:,0].T
            xreal=torch.matmul(ur,EI_real)-torch.matmul(ui,EI_imag)
            ximag=torch.matmul(ur,EI_imag)+torch.matmul(ui,EI_real)

            tmp_real=torch.mul(Gnk_real,xreal)-torch.mul(Gnk_imag,ximag)
            tmp_imag=torch.mul(Gnk_real,ximag)+torch.mul(Gnk_imag,xreal)
            wgt_real=torch.sum(tmp_real,dim=3).squeeze(-1)
            wgt_imag=torch.sum(tmp_imag,dim=3).squeeze(-1)
            angle_wgt = torch.atan2(wgt_imag, wgt_real)
            angle_wgt = angle_wgt.view(bsz * self.A_num * self.dim0, -1)
            angle_wgt = angle_wgt.repeat(1, self.dim1).view(bsz, self.A_num, self.dim0, -1)
            comp_wgt_real=torch.cos(angle_wgt)
            comp_wgt_imag=-1*torch.sin(angle_wgt)
            x_real=Gnk_real*comp_wgt_real-Gnk_imag*comp_wgt_imag
            x_imag=Gnk_real*comp_wgt_imag+Gnk_imag*comp_wgt_real

        x = torch.cat((x_real, x_imag), 1)
        xreal,ximag = self.input_layer3(x)

        # xreal = xreal.squeeze(-2).view(bsz, -1)
        # ximag = ximag.squeeze(-2).view(bsz, -1)
        # xreal=xreal.view(bsz, self.A_num, self.dim0, -1)
        # ximag = ximag.view(bsz, self.A_num, self.dim0, -1)
        # x=torch.sqrt(torch.pow(xreal,2)+torch.pow(ximag,2))

        xreal = xreal.squeeze(-1).view(bsz, -1)
        ximag = ximag.squeeze(-1).view(bsz, -1)
        xabs = torch.sqrt(torch.pow(xreal, 2) + torch.pow(ximag, 2))
        xreal = xreal / torch.max(xabs, dim=1).values.repeat(xabs.size(1), 1).T
        ximag = ximag / torch.max(xabs, dim=1).values.repeat(xabs.size(1), 1).T
        xreal = xreal.view(bsz, self.A_num, self.dim0, -1)
        ximag = ximag.view(bsz, self.A_num, self.dim0, -1)
        x = torch.sqrt(torch.pow(xreal, 2) + torch.pow(ximag, 2))

        for i in range(self.n_layer):
            res_x = self.mod2[i](x)
            x = x + res_x
            x = self.activate_layer2[i](x)
        x = self.out_layer(x)
        x = x.squeeze(-3)
        return x


class FrequencyRepresentationModule_2D_v0512(nn.Module):
    def __init__(self, params=None):
        super().__init__()
        self.A_num=2
        self.n_layer=params.fr_n_layers
        n_filters=1
        self.n_filter=n_filters
        self.dim0 = params.fr_inner_dim_0
        self.dim1 = params.fr_inner_dim_1
        kernel_size=3
        self.input_layer =ComplexConv2d2(1, params.fr_inner_dim_1*self.A_num, kernel_size=(1,params.signal_dim_1), padding=0, bias=False)
        self.concat_pre=ComplexConv2d2(self.A_num,1,kernel_size=(1,1),padding=0,bias=False)
        self.input_layer3 = ComplexConv2d2(self.n_filter, params.fr_inner_dim_0 * self.A_num,
                                          kernel_size=( params.signal_dim_0,1), padding=0, bias=False)

        concat=[]
        for i in range(params.fr_n_layers):
            concat+=[ComplexConv2d2(self.A_num,1,kernel_size=(1,1),padding=0,bias=False)]
        self.concat=nn.Sequential(*concat)

        complex_mod1=[]
        for i in range(2*params.fr_n_layers):
            complex_mod1+=[ComplexConv2d2(n_filters,n_filters*params.signal_dim_0,kernel_size=(params.signal_dim_0,1),padding=0,bias=False)]
        self.complex_mod1=nn.Sequential(*complex_mod1)

        mod2=[]
        for i in range(params.fr_n_layers):
            tmp2 = []
            tmp2 += [
                nn.Conv2d(self.A_num, self.A_num, kernel_size=kernel_size, padding=kernel_size//2,
                                bias=False,padding_mode='circular'),
                nn.BatchNorm2d(self.A_num),
                nn.ReLU(),
                nn.Conv2d(self.A_num, self.A_num, kernel_size=kernel_size, padding=kernel_size//2,
                              bias=False,padding_mode='circular'),
                nn.BatchNorm2d(self.A_num),
            ]
            mod2+= [nn.Sequential(*tmp2)]
        self.mod2=nn.Sequential(*mod2)
        activate_layer2 = []
        for i in range(params.fr_n_layers):
            activate_layer2+=[nn.ReLU()]
        self.activate_layer2=nn.Sequential(*activate_layer2)
        self.out_layer = nn.ConvTranspose2d(self.A_num, 1, kernel_size=params.fr_kernel_out, stride=params.fr_upsampling,
                                            padding=2, output_padding=1, bias=False)

    def forward(self, inp):
        bsz = inp.size(0)
        xreal,ximag = self.input_layer(inp)
        xreal=xreal.squeeze(-1).view(bsz,self.A_num,self.dim1,self.dim0)
        ximag = ximag.squeeze(-1).view(bsz, self.A_num, self.dim1, self.dim0)
        xcp_real=xreal.permute([0, 1, 3, 2])
        xcp_imag=ximag.permute([0,1,3,2])

        xcp = torch.cat((xcp_real, xcp_imag), 1)
        xcp_real, xcp_imag = self.concat_pre(xcp)

        x_real=xcp_real
        x_imag=xcp_imag
        Gnk_real=x_real
        Gnk_imag=x_imag

        for i in range(self.n_layer):
            x = torch.cat((x_real, x_imag), 1)
            x_real, x_imag = self.complex_mod1[i](x)
            x_real = x_real.squeeze(-2)
            x_imag = x_imag.squeeze(-2)
            x_real = x_real.view(bsz, self.n_filter, self.dim0, -1)
            x_imag = x_imag.view(bsz, self.n_filter, self.dim0, -1)
            Iqk_real1 = x_real
            Iqk_Imag1 = x_imag

            x_real, x_imag = self.complex_mod1[i+self.n_layer](x)
            x_real = x_real.squeeze(-2)
            x_imag = x_imag.squeeze(-2)
            x_real = x_real.view(bsz, self.n_filter, self.dim0, -1)
            x_imag = x_imag.view(bsz, self.n_filter, self.dim0, -1)
            Iqk_real2 = x_real
            Iqk_Imag2 = x_imag

            Iqk_abs=(torch.pow(Iqk_real1,2)+torch.pow(Iqk_Imag1,2))
            EI_real=(1+torch.log(Iqk_abs))*Iqk_real1
            EI_imag=(1+torch.log(Iqk_abs))*(-1)*Iqk_Imag1
            ur=self.complex_mod1[i].conv_r.weight[:,0,:,0].T
            ui=self.complex_mod1[i].conv_i.weight[:,0,:,0].T
            xreal=torch.matmul(ur,EI_real)-torch.matmul(ui,EI_imag)
            ximag=torch.matmul(ur,EI_imag)+torch.matmul(ui,EI_real)
            tmp_real=torch.mul(Gnk_real,xreal)-torch.mul(Gnk_imag,ximag)
            tmp_imag=torch.mul(Gnk_real,ximag)+torch.mul(Gnk_imag,xreal)
            wgt_real=torch.sum(tmp_real,dim=3).squeeze(-1)
            wgt_imag=torch.sum(tmp_imag,dim=3).squeeze(-1)
            angle_wgt = torch.atan2(wgt_imag, wgt_real)
            angle_wgt = angle_wgt.view(bsz * self.n_filter * self.dim0, -1)
            angle_wgt = angle_wgt.repeat(1, self.dim1).view(bsz, self.n_filter, self.dim0, -1)
            comp_wgt_real=torch.cos(angle_wgt)
            comp_wgt_imag=-1*torch.sin(angle_wgt)
            x_real1=Gnk_real*comp_wgt_real-Gnk_imag*comp_wgt_imag
            x_imag1=Gnk_real*comp_wgt_imag+Gnk_imag*comp_wgt_real

            Iqk_abs=(torch.pow(Iqk_real2,2)+torch.pow(Iqk_Imag2,2))
            EI_real=(1+torch.log(Iqk_abs))*Iqk_real2
            EI_imag=(1+torch.log(Iqk_abs))*(-1)*Iqk_Imag2
            ur=self.complex_mod1[i+self.n_layer].conv_r.weight[:,0,:,0].T
            ui=self.complex_mod1[i+self.n_layer].conv_i.weight[:,0,:,0].T
            xreal=torch.matmul(ur,EI_real)-torch.matmul(ui,EI_imag)
            ximag=torch.matmul(ur,EI_imag)+torch.matmul(ui,EI_real)
            tmp_real=torch.mul(Gnk_real,xreal)-torch.mul(Gnk_imag,ximag)
            tmp_imag=torch.mul(Gnk_real,ximag)+torch.mul(Gnk_imag,xreal)
            wgt_real=torch.sum(tmp_real,dim=3).squeeze(-1)
            wgt_imag=torch.sum(tmp_imag,dim=3).squeeze(-1)
            angle_wgt = torch.atan2(wgt_imag, wgt_real)
            angle_wgt = angle_wgt.view(bsz * self.n_filter * self.dim0, -1)
            angle_wgt = angle_wgt.repeat(1, self.dim1).view(bsz, self.n_filter, self.dim0, -1)
            comp_wgt_real=torch.cos(angle_wgt)
            comp_wgt_imag=-1*torch.sin(angle_wgt)
            x_real2=Gnk_real*comp_wgt_real-Gnk_imag*comp_wgt_imag
            x_imag2=Gnk_real*comp_wgt_imag+Gnk_imag*comp_wgt_real
            x_real=torch.cat((x_real1,x_real2),1)
            x_imag=torch.cat((x_imag1,x_imag2),1)
            xconcat=torch.cat((x_real,x_imag),1)

            x_real,x_imag=self.concat[i](xconcat)



        x = torch.cat((x_real, x_imag), 1)
        xreal,ximag = self.input_layer3(x)

        # xreal = xreal.squeeze(-2).view(bsz, -1)
        # ximag = ximag.squeeze(-2).view(bsz, -1)
        # xreal=xreal.view(bsz, self.A_num, self.dim0, -1)
        # ximag = ximag.view(bsz, self.A_num, self.dim0, -1)
        # x=torch.sqrt(torch.pow(xreal,2)+torch.pow(ximag,2))

        xreal = xreal.squeeze(-1).view(bsz, -1)
        ximag = ximag.squeeze(-1).view(bsz, -1)
        xabs = torch.sqrt(torch.pow(xreal, 2) + torch.pow(ximag, 2))
        xreal = xreal / torch.max(xabs, dim=1).values.repeat(xabs.size(1), 1).T
        ximag = ximag / torch.max(xabs, dim=1).values.repeat(xabs.size(1), 1).T
        xreal = xreal.view(bsz, self.A_num, self.dim0, -1)
        ximag = ximag.view(bsz, self.A_num, self.dim0, -1)
        x = torch.sqrt(torch.pow(xreal, 2) + torch.pow(ximag, 2))

        for i in range(self.n_layer):
            res_x = self.mod2[i](x)
            x = x + res_x
            x = self.activate_layer2[i](x)
        x = self.out_layer(x)
        x = x.squeeze(-3)
        return x

class FrequencyRepresentationModule_2D_v0519(nn.Module):
    def __init__(self, params=None):
        super().__init__()
        self.A_num=4
        self.n_layer=params.fr_n_layers
        n_filters=1
        self.n_filter=n_filters
        self.dim0 = params.fr_inner_dim_0
        self.dim1 = params.fr_inner_dim_1
        kernel_size=3
        self.input_layer =ComplexConv2d2(1, params.fr_inner_dim_1*self.A_num, kernel_size=(1,params.signal_dim_1), padding=0, bias=False)
        self.complex_conv = ComplexConv2d2(self.A_num, self.A_num, kernel_size=(params.signal_dim_1, 1), bias=False)
        self.concat_pre=ComplexConv2d2(self.A_num,1,kernel_size=(1,1),padding=0,bias=False)
        self.input_layer3 = ComplexConv2d2(self.n_filter, params.fr_inner_dim_0 * self.A_num,
                                          kernel_size=( params.signal_dim_0,1), padding=0, bias=False)

        concat=[]
        for i in range(params.fr_n_layers//2):
            concat+=[ComplexConv2d2(2,1,kernel_size=(1,1),padding=0,bias=False)]
        self.concat=nn.Sequential(*concat)

        complex_mod1=[]
        for i in range(2*params.fr_n_layers//2):
            complex_mod1+=[ComplexConv2d2(n_filters,n_filters*params.signal_dim_0,kernel_size=(params.signal_dim_0,1),padding=0,bias=False)]
        self.complex_mod1=nn.Sequential(*complex_mod1)

        mod2=[]
        for i in range(params.fr_n_layers//2):
            tmp2 = []
            tmp2 += [
                nn.Conv2d(self.A_num, self.A_num, kernel_size=kernel_size, padding=kernel_size//2,
                                bias=False,padding_mode='circular'),
                nn.BatchNorm2d(self.A_num),
                nn.ReLU(),
                nn.Conv2d(self.A_num, self.A_num, kernel_size=kernel_size, padding=kernel_size//2,
                              bias=False,padding_mode='circular'),
                nn.BatchNorm2d(self.A_num),
            ]
            mod2+= [nn.Sequential(*tmp2)]
        self.mod2=nn.Sequential(*mod2)
        activate_layer2 = []
        for i in range(params.fr_n_layers//2):
            activate_layer2+=[nn.ReLU()]
        self.activate_layer2=nn.Sequential(*activate_layer2)
        self.out_layer = nn.ConvTranspose2d(self.A_num, 1, kernel_size=params.fr_kernel_out, stride=params.fr_upsampling,
                                            padding=2, output_padding=1, bias=False)

    def forward(self, inp):
        bsz = inp.size(0)
        xreal,ximag = self.input_layer(inp)
        xreal=xreal.squeeze(-1).view(bsz,self.A_num,self.dim1,self.dim0)
        ximag = ximag.squeeze(-1).view(bsz, self.A_num, self.dim1, self.dim0)
        xcp_real=xreal.permute([0, 1, 3, 2])
        xcp_imag=ximag.permute([0,1,3,2])

        x1_real = xreal[:, :, :, 0:self.dim0 - 1]
        x1_imag_conj = -1 * ximag[:, :, :, 0:self.dim0 - 1]
        x2_real = xreal[:, :, :, 1:self.dim0]
        x2_imag = ximag[:, :, :, 1:self.dim0]
        x_new_real = x1_real * x2_real - x1_imag_conj * x2_imag
        x_new_imag = x1_real * x2_imag + x1_imag_conj * x2_real
        x_new = torch.cat((x_new_real, x_new_imag), 1)
        compx_real, compx_imag = self.complex_conv(x_new)
        compx_real = compx_real.squeeze(-2)
        compx_imag = compx_imag.squeeze(-2)
        angle_compx = torch.atan2(compx_imag, compx_real)
        angle_compx2 = torch.zeros([bsz, self.A_num, self.dim0]).to(inp.device)
        angle_compx2[:, :, 1:self.dim0] = angle_compx
        angle_compx = torch.cumsum(angle_compx2, dim=2).view(bsz * self.A_num * self.dim0, -1)
        angle_compx = angle_compx.repeat(1, self.dim1).view(bsz, self.A_num, self.dim0, -1)
        compx_real = torch.cos(angle_compx)
        compx_imag_conj = -1 * torch.sin(angle_compx)
        x_real = xcp_real * compx_real - xcp_imag * compx_imag_conj
        x_imag = xcp_real * compx_imag_conj + xcp_imag * compx_real

        xcp = torch.cat((x_real, x_imag), 1)
        xcp_real, xcp_imag = self.concat_pre(xcp)

        x_real=xcp_real
        x_imag=xcp_imag
        Gnk_real=x_real
        Gnk_imag=x_imag

        for i in range(self.n_layer//2):
            x = torch.cat((x_real, x_imag), 1)
            x_real, x_imag = self.complex_mod1[i](x)
            x_real = x_real.squeeze(-2)
            x_imag = x_imag.squeeze(-2)
            x_real = x_real.view(bsz, self.n_filter, self.dim0, -1)
            x_imag = x_imag.view(bsz, self.n_filter, self.dim0, -1)
            Iqk_real1 = x_real
            Iqk_Imag1 = x_imag

            x_real, x_imag = self.complex_mod1[i+self.n_layer//2](x)
            x_real = x_real.squeeze(-2)
            x_imag = x_imag.squeeze(-2)
            x_real = x_real.view(bsz, self.n_filter, self.dim0, -1)
            x_imag = x_imag.view(bsz, self.n_filter, self.dim0, -1)
            Iqk_real2 = x_real
            Iqk_Imag2 = x_imag

            Iqk_abs=(torch.pow(Iqk_real1,2)+torch.pow(Iqk_Imag1,2))
            EI_real=(1+torch.log(Iqk_abs))*Iqk_real1
            EI_imag=(1+torch.log(Iqk_abs))*(-1)*Iqk_Imag1
            ur=self.complex_mod1[i].conv_r.weight[:,0,:,0].T
            ui=self.complex_mod1[i].conv_i.weight[:,0,:,0].T
            xreal=torch.matmul(ur,EI_real)-torch.matmul(ui,EI_imag)
            ximag=torch.matmul(ur,EI_imag)+torch.matmul(ui,EI_real)
            tmp_real=torch.mul(Gnk_real,xreal)-torch.mul(Gnk_imag,ximag)
            tmp_imag=torch.mul(Gnk_real,ximag)+torch.mul(Gnk_imag,xreal)
            wgt_real=torch.sum(tmp_real,dim=3).squeeze(-1)
            wgt_imag=torch.sum(tmp_imag,dim=3).squeeze(-1)
            angle_wgt = torch.atan2(wgt_imag, wgt_real)
            angle_wgt = angle_wgt.view(bsz * self.n_filter * self.dim0, -1)
            angle_wgt = angle_wgt.repeat(1, self.dim1).view(bsz, self.n_filter, self.dim0, -1)
            comp_wgt_real=torch.cos(angle_wgt)
            comp_wgt_imag=-1*torch.sin(angle_wgt)
            x_real1=Gnk_real*comp_wgt_real-Gnk_imag*comp_wgt_imag
            x_imag1=Gnk_real*comp_wgt_imag+Gnk_imag*comp_wgt_real

            Iqk_abs=(torch.pow(Iqk_real2,2)+torch.pow(Iqk_Imag2,2))
            EI_real=(1+torch.log(Iqk_abs))*Iqk_real2
            EI_imag=(1+torch.log(Iqk_abs))*(-1)*Iqk_Imag2
            ur=self.complex_mod1[i+self.n_layer//2].conv_r.weight[:,0,:,0].T
            ui=self.complex_mod1[i+self.n_layer//2].conv_i.weight[:,0,:,0].T
            xreal=torch.matmul(ur,EI_real)-torch.matmul(ui,EI_imag)
            ximag=torch.matmul(ur,EI_imag)+torch.matmul(ui,EI_real)
            tmp_real=torch.mul(Gnk_real,xreal)-torch.mul(Gnk_imag,ximag)
            tmp_imag=torch.mul(Gnk_real,ximag)+torch.mul(Gnk_imag,xreal)
            wgt_real=torch.sum(tmp_real,dim=3).squeeze(-1)
            wgt_imag=torch.sum(tmp_imag,dim=3).squeeze(-1)
            angle_wgt = torch.atan2(wgt_imag, wgt_real)
            angle_wgt = angle_wgt.view(bsz * self.n_filter * self.dim0, -1)
            angle_wgt = angle_wgt.repeat(1, self.dim1).view(bsz, self.n_filter, self.dim0, -1)
            comp_wgt_real=torch.cos(angle_wgt)
            comp_wgt_imag=-1*torch.sin(angle_wgt)
            x_real2=Gnk_real*comp_wgt_real-Gnk_imag*comp_wgt_imag
            x_imag2=Gnk_real*comp_wgt_imag+Gnk_imag*comp_wgt_real
            x_real=torch.cat((x_real1,x_real2),1)
            x_imag=torch.cat((x_imag1,x_imag2),1)
            xconcat=torch.cat((x_real,x_imag),1)
            x_real,x_imag=self.concat[i](xconcat)

        x = torch.cat((x_real, x_imag), 1)
        xreal,ximag = self.input_layer3(x)

        # xreal = xreal.squeeze(-2).view(bsz, -1)
        # ximag = ximag.squeeze(-2).view(bsz, -1)
        # xreal=xreal.view(bsz, self.A_num, self.dim0, -1)
        # ximag = ximag.view(bsz, self.A_num, self.dim0, -1)
        # x=torch.sqrt(torch.pow(xreal,2)+torch.pow(ximag,2))

        xreal = xreal.squeeze(-1).view(bsz, -1)
        ximag = ximag.squeeze(-1).view(bsz, -1)
        xabs = torch.sqrt(torch.pow(xreal, 2) + torch.pow(ximag, 2))
        xreal = xreal / torch.max(xabs, dim=1).values.repeat(xabs.size(1), 1).T
        ximag = ximag / torch.max(xabs, dim=1).values.repeat(xabs.size(1), 1).T
        xreal = xreal.view(bsz, self.A_num, self.dim0, -1)
        ximag = ximag.view(bsz, self.A_num, self.dim0, -1)
        x = torch.sqrt(torch.pow(xreal, 2) + torch.pow(ximag, 2))

        for i in range(self.n_layer//2):
            res_x = self.mod2[i](x)
            x = x + res_x
            x = self.activate_layer2[i](x)
        x = self.out_layer(x)
        x = x.squeeze(-3)
        return x


class FrequencyRepresentationModule_2D_v0523(nn.Module):
    def __init__(self, params=None):
        super().__init__()
        self.A_num=2
        self.n_layer=params.fr_n_layers
        n_filters=1
        self.n_filter=n_filters
        self.dim0 = params.fr_inner_dim_0
        self.dim1 = params.fr_inner_dim_1
        kernel_size=3
        self.input_layer =ComplexConv2d2(1, params.fr_inner_dim_1*self.A_num, kernel_size=(1,params.signal_dim_1), padding=0, bias=False)
        self.complex_conv = ComplexConv2d2(self.A_num, self.A_num, kernel_size=(params.signal_dim_1, 1), bias=False)
        self.concat_pre=ComplexConv2d2(self.A_num,1,kernel_size=(1,1),padding=0,bias=False)
        self.input_layer3 = ComplexConv2d2(self.n_filter, params.fr_inner_dim_0 * self.A_num,
                                          kernel_size=( params.signal_dim_0,1), padding=0, bias=False)

        concat=[]
        for i in range(params.fr_n_layers):
            concat+=[ComplexConv2d2(2,1,kernel_size=(1,1),padding=0,bias=False)]
        self.concat=nn.Sequential(*concat)

        complex_mod1=[]
        for i in range(2*params.fr_n_layers):
            complex_mod1+=[ComplexConv2d2(n_filters,n_filters*params.signal_dim_0,kernel_size=(params.signal_dim_0,1),padding=0,bias=False)]
        self.complex_mod1=nn.Sequential(*complex_mod1)

        mod2=[]
        for i in range(params.fr_n_layers):
            tmp2 = []
            tmp2 += [
                nn.Conv2d(self.A_num, self.A_num, kernel_size=kernel_size, padding=kernel_size//2,
                                bias=False,padding_mode='circular'),
                nn.BatchNorm2d(self.A_num),
                nn.ReLU(),
                nn.Conv2d(self.A_num, self.A_num, kernel_size=kernel_size, padding=kernel_size//2,
                              bias=False,padding_mode='circular'),
                nn.BatchNorm2d(self.A_num),
            ]
            mod2+= [nn.Sequential(*tmp2)]
        self.mod2=nn.Sequential(*mod2)
        activate_layer2 = []
        for i in range(params.fr_n_layers):
            activate_layer2+=[nn.ReLU()]
        self.activate_layer2=nn.Sequential(*activate_layer2)
        self.out_layer = nn.ConvTranspose2d(self.A_num, 1, kernel_size=params.fr_kernel_out, stride=params.fr_upsampling,
                                            padding=2, output_padding=1, bias=False)

    def forward(self, inp):
        bsz = inp.size(0)
        xreal,ximag = self.input_layer(inp)
        xreal=xreal.squeeze(-1).view(bsz,self.A_num,self.dim1,self.dim0)
        ximag = ximag.squeeze(-1).view(bsz, self.A_num, self.dim1, self.dim0)
        xcp_real=xreal.permute([0, 1, 3, 2])
        xcp_imag=ximag.permute([0,1,3,2])

        x1_real = xreal[:, :, :, 0:self.dim0 - 1]
        x1_imag_conj = -1 * ximag[:, :, :, 0:self.dim0 - 1]
        x2_real = xreal[:, :, :, 1:self.dim0]
        x2_imag = ximag[:, :, :, 1:self.dim0]
        x_new_real = x1_real * x2_real - x1_imag_conj * x2_imag
        x_new_imag = x1_real * x2_imag + x1_imag_conj * x2_real
        x_new = torch.cat((x_new_real, x_new_imag), 1)
        compx_real, compx_imag = self.complex_conv(x_new)
        compx_real = compx_real.squeeze(-2)
        compx_imag = compx_imag.squeeze(-2)
        angle_compx = torch.atan2(compx_imag, compx_real)
        angle_compx2 = torch.zeros([bsz, self.A_num, self.dim0]).to(inp.device)
        angle_compx2[:, :, 1:self.dim0] = angle_compx
        angle_compx = torch.cumsum(angle_compx2, dim=2).view(bsz * self.A_num * self.dim0, -1)
        angle_compx = angle_compx.repeat(1, self.dim1).view(bsz, self.A_num, self.dim0, -1)
        compx_real = torch.cos(angle_compx)
        compx_imag_conj = -1 * torch.sin(angle_compx)
        x_real = xcp_real * compx_real - xcp_imag * compx_imag_conj
        x_imag = xcp_real * compx_imag_conj + xcp_imag * compx_real

        xcp = torch.cat((x_real, x_imag), 1)
        xcp_real, xcp_imag = self.concat_pre(xcp)

        x_real=xcp_real
        x_imag=xcp_imag
        Gnk_real=x_real
        Gnk_imag=x_imag

        for i in range(self.n_layer):
            x = torch.cat((x_real, x_imag), 1)
            x_real, x_imag = self.complex_mod1[i](x)
            x_real = x_real.squeeze(-2)
            x_imag = x_imag.squeeze(-2)
            x_real = x_real.view(bsz, self.n_filter, self.dim0, -1)
            x_imag = x_imag.view(bsz, self.n_filter, self.dim0, -1)
            Iqk_real1 = x_real
            Iqk_Imag1 = x_imag

            x_real, x_imag = self.complex_mod1[i+self.n_layer](x)
            x_real = x_real.squeeze(-2)
            x_imag = x_imag.squeeze(-2)
            x_real = x_real.view(bsz, self.n_filter, self.dim0, -1)
            x_imag = x_imag.view(bsz, self.n_filter, self.dim0, -1)
            Iqk_real2 = x_real
            Iqk_Imag2 = x_imag

            Iqk_abs=(torch.pow(Iqk_real1,2)+torch.pow(Iqk_Imag1,2))
            EI_real=(1+torch.log(Iqk_abs))*Iqk_real1
            EI_imag=(1+torch.log(Iqk_abs))*(-1)*Iqk_Imag1
            ur=self.complex_mod1[i].conv_r.weight[:,0,:,0].T
            ui=self.complex_mod1[i].conv_i.weight[:,0,:,0].T
            xreal=torch.matmul(ur,EI_real)-torch.matmul(ui,EI_imag)
            ximag=torch.matmul(ur,EI_imag)+torch.matmul(ui,EI_real)
            tmp_real=torch.mul(Gnk_real,xreal)-torch.mul(Gnk_imag,ximag)
            tmp_imag=torch.mul(Gnk_real,ximag)+torch.mul(Gnk_imag,xreal)
            wgt_real=torch.sum(tmp_real,dim=3).squeeze(-1)
            wgt_imag=torch.sum(tmp_imag,dim=3).squeeze(-1)
            angle_wgt = torch.atan2(wgt_imag, wgt_real)
            angle_wgt = angle_wgt.view(bsz * self.n_filter * self.dim0, -1)
            angle_wgt = angle_wgt.repeat(1, self.dim1).view(bsz, self.n_filter, self.dim0, -1)
            comp_wgt_real=torch.cos(angle_wgt)
            comp_wgt_imag=-1*torch.sin(angle_wgt)
            x_real1=Gnk_real*comp_wgt_real-Gnk_imag*comp_wgt_imag
            x_imag1=Gnk_real*comp_wgt_imag+Gnk_imag*comp_wgt_real

            Iqk_abs=(torch.pow(Iqk_real2,2)+torch.pow(Iqk_Imag2,2))
            EI_real=(1+torch.log(Iqk_abs))*Iqk_real2
            EI_imag=(1+torch.log(Iqk_abs))*(-1)*Iqk_Imag2
            ur=self.complex_mod1[i+self.n_layer].conv_r.weight[:,0,:,0].T
            ui=self.complex_mod1[i+self.n_layer].conv_i.weight[:,0,:,0].T
            xreal=torch.matmul(ur,EI_real)-torch.matmul(ui,EI_imag)
            ximag=torch.matmul(ur,EI_imag)+torch.matmul(ui,EI_real)
            tmp_real=torch.mul(Gnk_real,xreal)-torch.mul(Gnk_imag,ximag)
            tmp_imag=torch.mul(Gnk_real,ximag)+torch.mul(Gnk_imag,xreal)
            wgt_real=torch.sum(tmp_real,dim=3).squeeze(-1)
            wgt_imag=torch.sum(tmp_imag,dim=3).squeeze(-1)
            angle_wgt = torch.atan2(wgt_imag, wgt_real)
            angle_wgt = angle_wgt.view(bsz * self.n_filter * self.dim0, -1)
            angle_wgt = angle_wgt.repeat(1, self.dim1).view(bsz, self.n_filter, self.dim0, -1)
            comp_wgt_real=torch.cos(angle_wgt)
            comp_wgt_imag=-1*torch.sin(angle_wgt)
            x_real2=Gnk_real*comp_wgt_real-Gnk_imag*comp_wgt_imag
            x_imag2=Gnk_real*comp_wgt_imag+Gnk_imag*comp_wgt_real
            x_real=torch.cat((x_real1,x_real2),1)
            x_imag=torch.cat((x_imag1,x_imag2),1)
            xconcat=torch.cat((x_real,x_imag),1)
            x_real,x_imag=self.concat[i](xconcat)

        x = torch.cat((x_real, x_imag), 1)
        xreal,ximag = self.input_layer3(x)

        # xreal = xreal.squeeze(-2).view(bsz, -1)
        # ximag = ximag.squeeze(-2).view(bsz, -1)
        # xreal=xreal.view(bsz, self.A_num, self.dim0, -1)
        # ximag = ximag.view(bsz, self.A_num, self.dim0, -1)
        # x=torch.sqrt(torch.pow(xreal,2)+torch.pow(ximag,2))

        xreal = xreal.squeeze(-1).view(bsz, -1)
        ximag = ximag.squeeze(-1).view(bsz, -1)
        xabs = torch.sqrt(torch.pow(xreal, 2) + torch.pow(ximag, 2))
        xreal = xreal / torch.max(xabs, dim=1).values.repeat(xabs.size(1), 1).T
        ximag = ximag / torch.max(xabs, dim=1).values.repeat(xabs.size(1), 1).T
        xreal = xreal.view(bsz, self.A_num, self.dim0, -1)
        ximag = ximag.view(bsz, self.A_num, self.dim0, -1)
        x = torch.sqrt(torch.pow(xreal, 2) + torch.pow(ximag, 2))

        for i in range(self.n_layer):
            res_x = self.mod2[i](x)
            x = x + res_x
            x = self.activate_layer2[i](x)
        x = self.out_layer(x)
        x = x.squeeze(-3)
        return x