clc
clear
close all
rng('default')

B=1e9;
fc=9e9;
fs=1e9;
C=3e8;
np=64;
PRI=1/50;
dv=C/2/fc/np/PRI;
dr=C/2/B;
TS=1/fs;


n_rwnd=256;      
TW=n_rwnd*TS; 



sweep_slope=B/TW;
unfs=fs/(n_rwnd/256);
unr=C*unfs/2/sweep_slope;
x_label=(0:unr/512:unr-unr/512);

unv=C/2/fc/PRI;
y_label=(0:unv/128:unv-unv/128)-unv/2;

r_init=0*unr;
r=[r_init];
v_init=0*unv;
v=0;
a=0;

SNR=[-6,-3,0,3,6,9,12,15];

%% 
                                                   
                         
ITER=100;
SNR_ITER=length(SNR);
RD_entropy=zeros(SNR_ITER,ITER);
FMEPC_entropy=zeros(SNR_ITER,ITER);
ADN_entropy=zeros(SNR_ITER,ITER);
PGA_entropy=zeros(SNR_ITER,ITER);
ADMM_entropy=zeros(SNR_ITER,ITER);
SREM_entropy=zeros(SNR_ITER,ITER);

RD_time=zeros(SNR_ITER,ITER);
FMEPC_time=zeros(SNR_ITER,ITER);
ADN_time=zeros(SNR_ITER,ITER);
PGA_time=zeros(SNR_ITER,ITER);
ADMM_time=zeros(SNR_ITER,ITER);
SREM_time=zeros(SNR_ITER,ITER);
minAngle=1;
maxAngle=5;
for i=1:length(SNR)
    for kk=1:ITER
        [i,kk]


        rotate_w_du=((maxAngle-minAngle)*rand()+minAngle)/np/PRI;
        rotate_w=rotate_w_du/180*pi;
        tgt_num=randi(50);
        x=unr*rand(1,tgt_num)*0.8+0.1*unr;
        ForUnv_pre=maxAngle/180*pi;
        y=unv/ForUnv_pre*(rand(1,tgt_num)-0.5)*0.8+0.1*unv;
        n_tgt=length(x);      
        tgt_num=length(x);
        amp=ones(1,tgt_num);    


        for j=1:np
            sig_rngate_ds=zeros(1,n_rwnd);
            for k = 1:n_tgt
                %旋转
                theta=atan2(y(k),x(k));
                RR(j,k)=sqrt(x(k)^2+y(k)^2)*cos(theta+rotate_w*(j-1)*PRI);
                %平动
                tgt_dist=r+v*((j-1)*PRI)+a*((j-1)*PRI)^2/2+RR(j,k);
                tau2=2*(tgt_dist)/C;
                sig_rngate_ds=sig_rngate_ds+amp(k)*exp(-1i*2*pi*sweep_slope*(0:n_rwnd-1)*TS*tau2)*exp(-1i*2*pi*fc*tau2);

            end
            sig_rngate_ds=10^(SNR(i)/20)*sig_rngate_ds/sqrt(mean(power(abs(sig_rngate_ds),2)));
            noise=wgn(1,n_rwnd,0,'complex');
            sig_rngate_cps(j,:)=sig_rngate_ds+noise;
        end
        sig_mc=sig_rngate_cps;

        %% Simulations
        M=np;
        v=0.1;
        a=-0.005;

        N=n_rwnd;
        comp=exp(-1i*2*pi*(v+1/2*a)*(0:M-1)).';%*ones(1,N);
        motion_phase1=2*pi*v*(0:M-1)+2*pi*a/2*(0:M-1).^2;
        motion_phase2=2*pi*0.5*sin(0.5*(0:M-1));
        motion_phase3=motion_phase1+motion_phase2;
        motion_phase4=2*pi*(rand(1,M));
        motion_conb=[motion_phase1;motion_phase2;motion_phase3;motion_phase4];

        motion=exp(-1i*motion_conb(4,:)).'*ones(1,size(sig_mc,2));
        sig=sig_mc.*motion;
        final=ifft(sig,512,2);

        %% RD
        tic
        ret=abs(fftshift(fft(ifft(sig,512,2),128,1)));
        toc
        ret=fftshift(ret,1);
        RD_entropy(i,kk)=myE(ret);
        RD_time(i,kk)=toc;
        %% FMEPC
        tic
        fmepc=ifft2(sig);
        [ret,E]=FMEPC(fmepc);
        toc
        ret=fftshift(abs(ret),2);
        FMEPC_entropy(i,kk)=myE(ret);
        FMEPC_time(i,kk)=toc;
        %% PGA
        
        s1=final.';
        nrn=size(s1,1);
        tic
        [pga]=PGA(s1);
        toc
        ret=fftshift(fft(pga,128,2));
        ret=fftshift(abs(ret.'),1);
        PGA_entropy(i,kk)=myE(ret);
        PGA_time(i,kk)=toc;

        %% ADMM(交替方向乘子法)
        tic
        final2=PhaseComDoppler(final,v,a,PRI,fc);
        M=size(final2,1);
        N=size(final2,2);
        H=final2;
        e=diag(ones(1,M));
        pre_a=zeros(M,N);
        rou=1;
        lambda=0.005*var(H(:));
        pre_z=zeros(M,N);
        pre_x=zeros(M,N);

        cur_x=fft((e'*H+rou*ifft(pre_z)-ifft(pre_a))/(1+rou));
        iter=0;

        while sqrt(sum(sum((cur_x-pre_x).^2)))/sqrt(sum(sum(pre_x.^2)))>1e-5
            iter=iter+1;
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
        toc
        ret=fftshift(abs(cur_x));
        ADMM_entropy(i,kk)=myE(ret);
        ADMM_time(i,kk)=toc;
        %% ADN
        final=sig./(sqrt(mean(power(abs(sig),2),2))*ones(1,size(sig,2)));
        realdata=real(final);
        imagdata=imag(final);
        if ~exist('matlab_real.h5','file')==0
            delete('matlab_real.h5')
        end

        if ~exist('matlab_imag.h5','file')==0
            delete('matlab_imag.h5')   
        end

        h5create('matlab_real.h5','/matlab_real',size(realdata));
        h5write('matlab_real.h5','/matlab_real',(realdata))
        h5create('matlab_imag.h5','/matlab_imag',size(imagdata));
        h5write('matlab_imag.h5','/matlab_imag',(imagdata))


        flag=system('curl -s 127.0.0.1:5108/');
        py_data = (h5read('python_data.h5','/python_data')).';
        tt = (h5read('cost.h5','/cost'));
        Sp_deep=rot90(py_data,2);
        ret=py_data;
        ADN_entropy(i,kk)=myE(ret);
        ADN_time(i,kk)=tt;

        %% deep
        final=sig./(sqrt(mean(power(abs(sig),2),2))*ones(1,size(sig,2)));
        realdata=real(final);
        imagdata=imag(final);
        if ~exist('matlab_real.h5','file')==0
            delete('matlab_real.h5')
        end

        if ~exist('matlab_imag.h5','file')==0
            delete('matlab_imag.h5')   
        end

        h5create('matlab_real.h5','/matlab_real',size(realdata));
        h5write('matlab_real.h5','/matlab_real',(realdata))
        h5create('matlab_imag.h5','/matlab_imag',size(imagdata));
        h5write('matlab_imag.h5','/matlab_imag',(imagdata))

        flag=system('curl -s 127.0.0.1:5107/');
        py_data = (h5read('python_data.h5','/python_data')).';
        tt = (h5read('cost.h5','/cost'));
        Sp_deep=rot90(py_data,2);
        ret=py_data;
        ret=fftshift((abs(ret)),2);
        SREM_entropy(i,kk)=myE(ret);
        SREM_time(i,kk)=tt;
    end
end

RD_entropy_mean=mean(RD_entropy,2);
FMEPC_entropy_mean=mean(FMEPC_entropy,2);
PGA_entropy_mean=mean(PGA_entropy,2);
ADMM_entropy_mean=mean(ADMM_entropy,2);
ADN_entropy_mean=mean(ADN_entropy,2);
SREM_entropy_mean=mean(SREM_entropy,2);
statistic_entropy=[RD_entropy_mean,FMEPC_entropy_mean,PGA_entropy_mean,ADMM_entropy_mean,ADN_entropy_mean,SREM_entropy_mean];

RD_time_mean=mean(RD_time,2);
FMEPC_time_mean=mean(FMEPC_time,2);
ADN_time_mean=mean(ADN_time,2);
PGA_time_mean=mean(PGA_time,2);
ADMM_time_mean=mean(ADMM_time,2);
SREM_time_mean=mean(SREM_time,2);
statistic_time=[RD_time_mean,FMEPC_time_mean,PGA_time_mean,ADMM_time_mean,ADN_time_mean,SREM_time_mean];

save simu_statistic.mat statistic_entropy statistic_time
load simu_statistic.mat
SNR=[-6,-3,0,3,6,9,12,15];
figure;
ha = tight_subplot(1,2,[.03 .06],[.2 .02],[.06 .02]);
axes(ha(1))
for col=1:size(statistic_entropy,2)
    plot(SNR,statistic_entropy(:,col),'--o','Linewidth',2);
    hold on;
end
xlabel({'\fontsize{16pt}\rm{SNR / dB}';'\fontsize{16pt}\rm{(a)}'})
ylabel('\fontsize{16pt}\rm{Image Entropy}')
xlim([-8 17])
grid on;
legend('RD','FMEPC','PGA','ADMM','2-D ADN','AFNet')
set(gca,'FontSize',15)

axes(ha(2))
for col=1:size(statistic_time,2)
    
    semilogy(SNR,statistic_time(:,col),'--o','Linewidth',2);
    hold on;
end
legend('RD','FMEPC','PGA','ADMM','2-D ADN','AFNet')
xlim([-8 17])
xlabel({'\fontsize{16pt}\rm{SNR / dB}';'\fontsize{16pt}\rm{(b)}'})
ylabel('\fontsize{16pt}\rm{Computational Cost}')
grid on
set(gca,'FontSize',15)