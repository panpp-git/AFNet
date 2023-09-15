function EchoSig=RefSignalGen(TW,BW,rnggate_min,n_rwnd,tgt_dist,FS)
%% ������ʧ��ο��ź�

C=3e8;
sweep_slope=BW/TW;                                        %��Ƶ��
TS=1/FS;
RNG_MAX=rnggate_min+(n_rwnd-1)*TS*C/2;
%==================================================================
%% Gnerate the echo      
t=linspace(2*rnggate_min/C,2*RNG_MAX/C,n_rwnd);               %���ţ�ʱ��
                                                          %����ǰ�� t=2*Rmin/C
                                                          %���ź��� t=2*Rmax/C                            
M=length(tgt_dist);                                       %Ŀ����Ŀ                                       
td=ones(M,1)*t-2*tgt_dist'/C*ones(1,n_rwnd);
Srt=exp(j*pi*sweep_slope*td.^2);%radar echo from point targets 
EchoSig=Srt;