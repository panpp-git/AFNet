function EchoSig=RefSignalGen(TW,BW,rnggate_min,n_rwnd,tgt_dist,FS)
%% 本地无失真参考信号

C=3e8;
sweep_slope=BW/TW;                                        %调频率
TS=1/FS;
RNG_MAX=rnggate_min+(n_rwnd-1)*TS*C/2;
%==================================================================
%% Gnerate the echo      
t=linspace(2*rnggate_min/C,2*RNG_MAX/C,n_rwnd);               %波门（时域）
                                                          %波门前沿 t=2*Rmin/C
                                                          %波门后沿 t=2*Rmax/C                            
M=length(tgt_dist);                                       %目标数目                                       
td=ones(M,1)*t-2*tgt_dist'/C*ones(1,n_rwnd);
Srt=exp(j*pi*sweep_slope*td.^2);%radar echo from point targets 
EchoSig=Srt;