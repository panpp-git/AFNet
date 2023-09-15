function [vars,idx]=sgood(dat)
%  function [vars,idx]=sgood(dat)
%  Function to calculate and sort the normalized amplitude variance
%  dat -- complex echo data after range alignment
%  nsum -- number of range bins whose normalized variance to be summed
%  father functions: sltbin, sltcas
%  son functions:  Nil
[nrn,nan]=size(dat);
amp=abs(dat);
for n=1:nrn
  dm=mean(amp(n,:));
  dvar=cov(amp(n,:));
  nvar(n)=dvar/(dvar+dm*dm+1e-6);%%???
end
[vars,idx]=sort(nvar);  %%将每个距离门的信号方差按小到大排列

%nan=256;
%dm=sum(amp(n,:))/nan;
%dvar=(abs(xr(160,:))*abs(xr(160,:).'))/nan-(sum(abs(xr(160,:)))/nan).^2;

%for n=1:nrn
%  dm=mean(amp(n,:));
%  d2m=mean(amp(n,;).*amp(n,:));
%  nvar(n)=(d2m-dm*dm)/(d2m*d2m);
%end



