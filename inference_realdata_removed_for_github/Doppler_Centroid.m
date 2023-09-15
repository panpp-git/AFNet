function [dataout,phase_estim]=Doppler_Centroid(datain,varlim)
%
% Reference:�øĽ��Ķ��������ĸ��ٷ�����ISAR�˶�����,���״��
%
% ԭ�����������ĸ��ٷ��е�����,�ǽ����ڻز��źŸ���������λ
% �ָ�����������ȳ˻����м�Ȩƽ����.
%
% varlim:ѡ����뵥Ԫʱ�趨�İ����һ��������ֵ
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
phase_estim = [0 cumsum(delta_phase)];   %%�ۻ���λ��
linear_coefs = polyfit(1:length(phase_estim), phase_estim, 1);  %%������λ���
phase_estim = unwrap(phase_estim - polyval(linear_coefs, 1:length(phase_estim)));  %%�����ڷ�λ�����λ�����ֵ
dataout = dataout .* repmat(exp(-1j*phase_estim), M, 1);