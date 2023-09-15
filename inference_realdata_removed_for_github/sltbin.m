function [dato,vars,len,idx]=sltbin(datin,varlim)
%[dato,vars,len]=sltbin(datin,varlim)
%  function to select range bins, whos normalized amplitude variance
%  (NAV) is smaller than 'varlim', for estimate the phase errors
%  varlim -- threshold of NAV.
%  dato-- output selected COMPLEX image
%  vars -- NAV of selected bins
%  father functions:  wmsa, casia
%  son functions:  sgood
[nrn,nan]=size(datin);
[var,idx]=sgood(datin);
if var(1)>varlim
  varlim=var(1);
end
len=0;
for n=1:nrn
  if var(n)<=varlim
    len=len+1;
    dato(len,:)=datin(idx(n),:);
  end
end;
len;
vars=var(1:len); %%找到方差最小的距离门

