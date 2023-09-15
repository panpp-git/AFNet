%���������ķ���λ����
%gj 2007 6 5

%% phase compasation
function R_abs=PhaseComDoppler(R_abs,vec,aec,PRI,FC)
% load R_evp.mat;

[N,M]=size(R_abs);
phase=zeros(1,M);
pref=R_abs(1,:);
% R_abs(1,:)=R_abs(1,:).*conj(R_abs(1,:));
for k=1:N-1
   Delta=conj(pref).*R_abs(k+1,:);                             %��λ��
   phase=sum(Delta)/abs(sum(Delta));                           %��Ȩ��һ��
   R_abs(k+1,:)=R_abs(k+1,:).*conj(phase);                            %����
   if(k<8)
        pref=pref+R_abs(k+1,:);
   else
        pref=pref+R_abs(k+1,:)-R_abs(k-7,:);                   %�ο���������
   end
   ret(k)=phase;
end

% 
% [N,M]=size(R_abs);
% phase=zeros(1,M);
% pref=R_abs(1,:);
% for k=1:N-1
%     tau=aec*(PRI*k)^2/2+vec*(PRI*k);
%     comp=exp(-1i*2*pi*FC*2*tau/3e8);
% %     Delta=conj(pref).*R_abs(k+1,:);                             %��λ��
% %     phase=sum(Delta)/abs(sum(Delta));                           %��Ȩ��һ��
%     R_abs(k+1,:)=R_abs(k+1,:).*conj(comp);                            %����
% end