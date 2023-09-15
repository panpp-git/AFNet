clc
clear
close all

load b727r.mat
X=X.';
X=ifft(X,[],2);
X=[X,zeros(128,128)];
fsz=13;
dr=3e8/2/450e6;
r=size(X,2)*dr;
r_label=0:r/512:r-r/512;
sig=(X(1+2:64+2,1:end));
RPP=sig;
final=ifft(RPP,512,2);
ha = tight_subplot(2,3,[.052 .006],[.13 .02],[.07 .02]);
%% simulated motion
% M=64;
% N=256;
% v=(rand()-0.5)*6000;
% a=(rand()-0.5)*2000/M;
% motion=(exp(-1i*2*pi*v*(0:M-1)).*exp(-1i*2*pi*a*(0:M-1).^2)).'*ones(1,N);
% RPP=RPP.*motion;

%% RD
axes(ha(1))
spc=((abs(ifft(fft(sig,128,1),512,2))));
eval(['save boeing_rd',' spc r_label']) ;
imagesc(r_label,1:size(spc,1),fftshift(spc,1));
xlabel({'\fontsize{7pt}\rm{(a)}'})
ylabel({'\fontsize{7pt}\rm{Cross-range Cell}'})
set(gca,'FontSize',6)
%% FMEPC
axes(ha(2))
fmepc=fft(ifft(sig,[],2),[],1);
[ret,E]=FMEPC(fmepc);
ret=fftshift(abs(flipud(ret)),1);
eval(['save boeing_fmepc',' ret r_label']) ;
imagesc(r_label,1:size(ret,1),ret);
xlabel({'\fontsize{7pt}\rm{(b)}'})
set(gca,'FontSize',6)
%% MSA
% axes(ha(3))
% s1=final;
% s3=Compensate(s1);
% ret=fftshift(abs(fft(s3,128,1)),1);
% imagesc(r_label,1:size(ret,1),ret);
% xlabel({'\fontsize{7pt}\rm{(c)}'})
% set(gca,'FontSize',6)
%% PGA
axes(ha(3))
final=ifft(RPP,512,2);
s1=final.';
nrn=size(s1,1);
tic
[pga]=PGA(s1);
toc
ret=fftshift(fft(pga,128,2),2);
eval(['save boeing_pga',' ret r_label']) ;
imagesc(r_label,1:size(ret.',1),abs(ret.'));
xlabel({'\fontsize{7pt}\rm{(c)}'})
set(gca,'FontSize',6)
%% ADMM
axes(ha(4))
M=size(final,1);
N=size(final,2);
H=final;
e=diag(ones(1,M));
pre_a=zeros(M,N);
rou=1;
lambda=0.005*var(H(:));
pre_z=zeros(M,N);
pre_x=zeros(M,N);

cur_x=fft((e'*H+rou*ifft(pre_z)-ifft(pre_a))/(1+rou));
iter=0;
while sqrt(sum(sum((cur_x-pre_x).^2)))/sqrt(sum(sum(pre_x.^2)))>1e-5
    iter=iter+1
    x1=cur_x+pre_a/rou;
    x2=lambda/rou;
    cur_z=(x1/sum(sum(abs(x1))))*max(sum(sum(abs(x1)))-x2,0);
    cur_a=pre_a+rou*(cur_x-cur_z);
    e=zeros(M,1);
    for ii=1:M
        H1=zeros(M,N);
        H1(ii,:)=H(ii,:);
        e(ii)=exp(1i*angle(sum(sum((1+(abs(cur_x)).^2).*conj(cur_x).*fft(H1/(1+rou))))));
    end
    e=diag(e);
    pre_z=cur_z;
    pre_a=cur_a;
    pre_x=cur_x;
    cur_x=fft((e'*H+rou*ifft(pre_z)-ifft(pre_a))/(1+rou));
end
ret=fftshift(abs(cur_x),1);
eval(['save boeing_admm',' ret r_label']) ;
imagesc(r_label,1:size(ret,1),ret)
xlabel({'\fontsize{7pt}\rm{Range/m}','\fontsize{7pt}\rm{(d)}'})
ylabel('\fontsize{7pt}\rm{Cross-range Cell}')
set(gca,'FontSize',6)
%% 2D-ADN
axes(ha(5))
RPP=RPP./(sqrt(mean(power(abs(RPP),2),2))*ones(1,size(RPP,2)));
if ~exist('matlab_real.h5','file')==0
    delete('matlab_real.h5')
end

if ~exist('matlab_imag.h5','file')==0
    delete('matlab_imag.h5')   
end

h5create('matlab_real.h5','/matlab_real',size(RPP));
h5write('matlab_real.h5','/matlab_real',real(RPP));
h5create('matlab_imag.h5','/matlab_imag',size(RPP));
h5write('matlab_imag.h5','/matlab_imag',imag(RPP));


flag=system('curl -s 127.0.0.1:5108/');
py_data = (h5read('python_data.h5','/python_data')).';

ret=(py_data);
ret=((abs(ret)/max(max(abs(ret)))));
eval(['save boeing_adn',' ret r_label']) ;
imagesc(r_label,1:size(ret,1),fftshift(ret,1));
xlabel({'\fontsize{7pt}\rm{Range/m}';'\fontsize{7pt}\rm{(e)}'})
set(gcf,'Position',[0,0,900,400])
set(gca,'FontSize',6)
%% SREM-AFNet
axes(ha(6))
RPP=RPP./(sqrt(mean(power(abs(RPP),2),2))*ones(1,size(RPP,2)));
if ~exist('matlab_real.h5','file')==0
    delete('matlab_real.h5')
end

if ~exist('matlab_imag.h5','file')==0
    delete('matlab_imag.h5')   
end

h5create('matlab_real.h5','/matlab_real',size(RPP));
h5write('matlab_real.h5','/matlab_real',real(RPP));
h5create('matlab_imag.h5','/matlab_imag',size(RPP));
h5write('matlab_imag.h5','/matlab_imag',imag(RPP));


flag=system('curl -s 127.0.0.1:5107/');
py_data = (h5read('python_data.h5','/python_data')).';

ret=(py_data);
ret=((abs(ret)/max(max(abs(ret)))));
eval(['save boeing_deep',' ret r_label']) ;
imagesc(r_label,1:size(ret,1),(ret));
xlabel({'\fontsize{7pt}\rm{Range/m}';'\fontsize{7pt}\rm{(f)}'})
set(gcf,'Position',[0,0,900,400])
set(gca,'FontSize',6)

set(ha(2),'yticklabel','')
set(ha(3),'yticklabel','')
set(ha(5),'yticklabel','')
set(ha(6),'yticklabel','')
set(ha(1),'xticklabel','')
set(ha(2),'xticklabel','')
set(ha(3),'xticklabel','')
