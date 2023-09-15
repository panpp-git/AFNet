function EchoSig=EchoSignalGen(TW,BW,RNG_MIN,n_rwnd,tgt_dist,amp,FC,TS)
%% LFM EchoSignal Generation
%CΪ����
%TWΪ�ź�����
%BWΪLFM�źŴ���
%sweep_slopeΪ��Ƶ��
%RNG_MINΪ����ǰ��
%RNG_MAXΪ���ź���
%r_rwndΪ�ܵ���
%tgt_distΪĿ��λ��
%ampΪĿ��ز�����
%FCΪ��Ƶ
%TSΪ�������
%==================================================================
%% Parameter
% C=299792458;                                            %����
C=3e8;
sweep_slope=BW/TW;                                        %��Ƶ��

RNG_MAX=RNG_MIN+(n_rwnd-1)*TS*C/2;
%==================================================================
%% Gnerate the echo      
t=linspace(2*RNG_MIN/C,2*RNG_MAX/C,n_rwnd);               %���ţ�ʱ��
                                                          %����ǰ�� t=2*Rmin/C
                                                          %���ź��� t=2*Rmax/C                            
M=length(tgt_dist);                                       %Ŀ����Ŀ                                       
td=ones(M,1)*t-2*tgt_dist'/C*ones(1,n_rwnd);
Srt=amp*((ones(n_rwnd,1)*exp(-j*2*pi*FC*2*tgt_dist/C)).'.*exp(j*pi*sweep_slope*td.^2).*(abs(td)<TW/2));%radar echo from point targets 
EchoSig=Srt;
%==================================================================

