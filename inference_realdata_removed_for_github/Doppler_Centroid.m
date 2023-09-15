function [dataout,phase_estim]=Doppler_Centroid(datain,varlim)
%
% Reference:用改进的多普勒中心跟踪法进行ISAR运动补偿,朱兆达等
%
% 原理：多普勒中心跟踪法中的作法,是将相邻回波信号各距离门相位
% 差复指数函数按幅度乘积进行加权平均。.
%
% varlim:选择距离单元时设定的包络归一化方差阈值
%
%

[M,N] = size(datain);
dataout = datain;
if nargin < 2, varlim = 0.3; end
% varlim = 0.2;
[datasl,vars,len,idx] = sltbin(datain,varlim);
clear datain
datain = datasl;
adjacent_echo = conj(datain(:,1:end-1)) .* datain(:, 2:end);
adj_sum = sum(abs(adjacent_echo),1);
for i = 1:length(adj_sum)
    delta_phase(i) = angle(sum(adjacent_echo(:,i)/adj_sum(i),1));
end
phase_estim = [0 cumsum(delta_phase)];   %%累积相位差
linear_coefs = polyfit(1:length(phase_estim), phase_estim, 1);  %%估计相位差函数
phase_estim = unwrap(phase_estim - polyval(linear_coefs, 1:length(phase_estim)));  %%求相邻方位向的相位差估计值
dataout = dataout .* repmat(exp(-1j*phase_estim), M, 1);