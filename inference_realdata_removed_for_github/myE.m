function [out]=myE(sig)
    D=(abs(sig)).^2/(sum(sum((abs(sig)).^2)))+1e-20;
    out=-sum(sum(D.*log(D)));
end