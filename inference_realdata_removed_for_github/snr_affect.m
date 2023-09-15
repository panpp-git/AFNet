clc
clear
close all
rng('default')
bsz=1;
for iter=1:bsz
    iter
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
    
    minAngle=5;
    maxAngle=5;

    rotate_w_du=((maxAngle-minAngle)*rand()+minAngle)/np/PRI
    rotate_w=rotate_w_du/180*pi;

    
    sweep_slope=B/TW;
    unfs=fs/(n_rwnd/256);
    unr=C*unfs/2/sweep_slope;
    x_label=(0:unr/512:unr-unr/512);
   
    unv=C/2/fc/PRI;
    y_label=(0:unv/128:unv-unv/128)-unv/2;
    SNR=[-6,-3,0,3];

    r_init=0*unr;
    r=[r_init];
    v_init=0*unv;
    v=0;
    a=0;

    tgt_num=20;
    x=unr*rand(1,tgt_num)*0.8+0.1*unr;
    ForUnv_pre=maxAngle/180*pi;
    y=unv/ForUnv_pre*(rand(1,tgt_num)-0.5)*0.8+0.1*unv;
    n_tgt=length(x);          
    amp=1*rand(1,n_tgt);
  
    
    %% 


    ha = tight_subplot(6,4,[.025 .03],[.07 .02],[.07 .02]);
    for i=1:length(SNR)
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
        UNR=unr;
        UNV=unv;
        comp=exp(-1i*2*pi*(v+1/2*a)*(0:M-1)).';%*ones(1,N);
        motion_phase1=2*pi*v*(0:M-1)+2*pi*a/2*(0:M-1).^2;
        motion_phase2=2*pi*0.5*sin(0.5*(0:M-1));
        motion_phase3=motion_phase1+motion_phase2;
        motion_phase4=2*pi*(rand(1,M));
        motion_conb=[motion_phase1;motion_phase2;motion_phase3;motion_phase4];

        motion=exp(-1i*motion_conb(1,:)).'*ones(1,size(sig_mc,2));
        sig=sig_mc.*motion;
        final=ifft(sig,512,2);

        %% RD
        ret=abs(fftshift(fft(ifft(sig,512,2),128,1)));
        x_label=(0:UNR/size(ret,2):UNR-UNR/size(ret,2));
        y_label=(0:UNV/size(ret,1):UNV-UNV/size(ret,2))-UNV/2;
        axes(ha(i));
        ret=abs(ret)/max(max(abs(ret)));
        zlim_low=-40;
        clims=[zlim_low 0];
        idx=i;
        eval(['save simu_snr_rd',num2str(i),' ret clims x_label y_label idx']) ;
        imagesc(x_label,y_label,20*log10(fftshift(ret)),clims);
        xlabel({'\fontsize{4pt}\rm{Range/m}'})
        if i==1
            ylabel({'\fontsize{8pt}\bf{RD}';'\fontsize{4pt}\rm{Doppler/Hz}'})
            title({'\fontsize{8pt}\bf{-6 dB}'})
        else
            ylabel({'\fontsize{4pt}\rm{Doppler/Hz}'})
            if i==2
                title({'\fontsize{8pt}\bf{-3 dB}'})
            end
            if i==3
                title({'\fontsize{8pt}\bf{0 dB}'})
            end
            if i==4
                title({'\fontsize{8pt}\bf{3 dB}'})
            end
        end
        set(gca,'FontSize',4)
          %% FMEPC
        fmepc=ifft2(sig);
        [ret,E]=FMEPC(fmepc);
        x_label=(0:UNR/size(ret,2):UNR-UNR/size(ret,2));
        y_label=(0:UNV/size(ret,1):UNV-UNV/size(ret,2))-UNV/2;
        axes(ha(4+i));
        ret=abs(ret)/max(max(abs(ret)));
        idx=i+4;
        eval(['save simu_snr_fmepc',num2str(i),' ret clims x_label y_label idx']) ;
        if 0
            imagesc(x_label,y_label,20*log10(fftshift(abs(ret))),clims);
        else
            imagesc(x_label,y_label,20*log10(fftshift(abs(ret),1)),clims);
        end
        xlabel({'\fontsize{4pt}\rm{Range/m}'})
        if i==1
            ylabel({'\fontsize{8pt}\bf{FMEPC}';'\fontsize{4pt}\rm{Doppler/Hz}'})
        else
            ylabel('\fontsize{4pt}\rm{Doppler/Hz}')
        end
        set(gca,'FontSize',4)
        %% MSA
    %     ret=Compensate(final);
    %     ret=fft(ret,128,1);
    %     axes(ha(8+i));
    %     x_label=(0:UNR/size(ret,2):UNR-UNR/size(ret,2));
    %     y_label=(0:UNV/size(ret,1):UNV-UNV/size(ret,2))-UNV/2;
    %     imagesc(x_label,y_label,fftshift(abs(ret),2));
    %     xlabel({'\fontsize{4pt}\rm{Range/m}'})
    %     if i==1
    %         ylabel({'\fontsize{8pt}\bf{MSA}';'\fontsize{4pt}\rm{Doppler/Hz}'})
    %     else
    %         ylabel('\fontsize{4pt}\rm{Doppler/Hz}')
    %     end
    %     set(gca,'FontSize',4)
        %% PGA
        s1=final.';
        nrn=size(s1,1);
        [pga]=PGA(s1);
        ret=fftshift(fft(pga,128,2));
        axes(ha(8+i));
        x_label=(0:UNR/size(ret,2):UNR-UNR/size(ret,2));
        y_label=(0:UNV/size(ret,1):UNV-UNV/size(ret,2))-UNV/2;
        ret=abs(ret)/max(max(abs(ret)));
        idx=i+8;
        eval(['save simu_snr_pga',num2str(i),' ret clims x_label y_label idx']) ;
        if 0
            imagesc(x_label,y_label,(abs(ret.')));
        else
       
            imagesc(x_label,y_label,20*log10(fftshift(abs(ret.'),2)),clims);
        end
        xlabel({'\fontsize{4pt}\rm{Range/m}'})
        if i==1
            ylabel({'\fontsize{8pt}\bf{PGA}';'\fontsize{4pt}\rm{Doppler/Hz}'})
        else
            ylabel('\fontsize{4pt}\rm{Doppler/Hz}')
        end
        set(gca,'FontSize',4)
        %% ADMM(交替方向乘子法)
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

        while sqrt(sum(sum((cur_x-pre_x).^2)))/sqrt(sum(sum(pre_x.^2)))>1e-9
            iter=iter+1
            x1=cur_x+pre_a/rou;
            x2=lambda/rou;
            cur_z=(x1/sum(sum(abs(x1))))*max(sum(sum(abs(x1)))-x2,0);
            cur_a=pre_a+rou*(cur_x-cur_z);
        %     e=diag(exp(1i*angle(sum(((1+(abs(cur_x)).^2).*conj(cur_x).*fft(H/(1+rou))),2))));
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

        x_label=(0:UNR/size(cur_x,2):UNR-UNR/size(cur_x,2));
        y_label=(0:UNV/size(cur_x,1):UNV-UNV/size(cur_x,2))-UNV/2;
        axes(ha(12+i));
        cur_x=abs(cur_x)/max(max(abs(cur_x)));
        idx=i+12;
        eval(['save simu_snr_admm',num2str(i),' cur_x clims x_label y_label idx']) ;
        imagesc(x_label,y_label,20*log10(fftshift(abs(cur_x),1)),clims);
        xlabel({'\fontsize{4pt}\rm{Range/m}'})
        if i==1
            ylabel({'\fontsize{8pt}\bf{ADMM}';'\fontsize{4pt}\rm{Doppler/Hz}'})
        else
            ylabel('\fontsize{4pt}\rm{Doppler/Hz}')
        end
        set(gca,'FontSize',4)
        %% 2D ADN
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
        x_label=(0:UNR/size(ret,2):UNR-UNR/size(ret,2));
        y_label=(0:UNV/size(ret,1):UNV-UNV/size(ret,2))-UNV/2;
        axes(ha((16+i)));
        ret=abs(ret)/max(max(abs(ret)));
        idx=i+16;
        eval(['save simu_snr_adn',num2str(i),' ret clims x_label y_label idx']) ;
        imagesc(x_label,y_label,20*log10(fftshift(abs(ret),1)),clims);
        xlabel({'\fontsize{4pt}\rm{Range/m}'})

        if i==1
            ylabel({'\fontsize{8pt}\bf{2-D ADN}';'\fontsize{4pt}\rm{Doppler/Hz}'})
        else
            ylabel('\fontsize{4pt}\rm{Doppler/Hz}')
        end
        set(gca,'FontSize',4)
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
        x_label=(0:UNR/size(ret,2):UNR-UNR/size(ret,2));
        y_label=(0:UNV/size(ret,1):UNV-UNV/size(ret,2))-UNV/2;
        axes(ha((20+i)));
        ret=abs(ret)/max(max(abs(ret)));
        idx=i+20;
        eval(['save simu_snr_deep',num2str(i),' ret clims x_label y_label idx']) ;
        imagesc(x_label,y_label,20*log10((abs(ret))),clims);
        xlabel({'\fontsize{4pt}\rm{Range/m}'})

        if i==1
            ylabel({'\fontsize{8pt}\bf{SREM-AFNet}';'\fontsize{4pt}\rm{Doppler/Hz}'})
        else
            ylabel('\fontsize{4pt}\rm{Doppler/Hz}')
        end
        set(gca,'FontSize',4)
        aa=1
    end
    set(gcf,'Position',[0,0,600,900])
end