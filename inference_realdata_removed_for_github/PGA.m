function [datain] = PGA(datain,varlim,iteration_num)
datain=Doppler_Centroid(datain);
if nargin < 2
    varlim = 0.3; 
    iteration_num=30;
end
for iii = 1:iteration_num
    
    [M,N] = size(datain);
    dataout = datain;%这里是整个的矩阵
    % varlim = 0.3;
    [datasl,vars,len,idx] = sltbin(datain,varlim);
    clear datain
    datain = datasl;%这里是挑选出来的矩阵
    phase_correction = ones(1,N);
    
    orig_img = fftshift(fft(datain,[],2),2);
    
    % Circular Shifting
    center_az_idx = ceil(N/2);
    [tmp maximum_az_idx] = max(abs(orig_img), [], 2);
    new_img = zeros(len,N);
    for i = 1:size(orig_img,1)
        new_img(i,:) = circshift(transpose(orig_img(i,:)), center_az_idx - maximum_az_idx(i));
    end
   % 
   
        window = 20*log10(sum(abs(new_img).^2, 1));
        scatter_power = window;
        db_down = 30;
        tmp=(min(window)+max(window))/2;
%         window_cutoff = max(window)-db_down;
        window_cutoff=tmp;
        leftidx = find(window(1:center_az_idx) - window_cutoff<0, 1, 'last' );
        rightidx = find(window(center_az_idx+1:end) - window_cutoff<0, 1, 'first');
        leftidx = leftidx+1;
        rightidx = rightidx + center_az_idx - 1;
       
    window = zeros(1,N);
    window(leftidx:rightidx) = 1;

    new_img = new_img.*repmat(window,len,1);
    new_data = ifft(fftshift(new_img,2),N,2);
    
%     S_phase=Ropee(new_data);
%     datain = dataout .* repmat(exp(-1j*S_phase), M, 1);
    [datainp,S_phase]=Doppler_Centroid(new_data,1);
    datain = dataout .* repmat(exp(-1j*S_phase), M, 1);
end


