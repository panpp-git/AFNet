% function RetData=Compensate(Data)
% %�������λ��׼
% [rows,cols]=size(Data);
% xw1=zeros(1,cols);
% xw2=zeros(1,cols);
% RetData(1,:)=Data(1,:);
% for k=2:rows
%     xw1=Data(k,:).*conj(RetData(k-1,:));%�������λز��������
%     xw2=abs(xw1);%
%     sum1=sum(xw1);%
%     sum2=sum(xw2);%
%     RetData(k,:)=Data(k,:)*conj(sum1/sum2);
% end

% function RetData=Compensate(Data)
% %�����Ե㷨����У׼
% [rows,cols]=size(Data);%�У���λ �У�����
% scatter=zeros(1,cols);
% for n=1:cols
%     ave=mean(abs(Data(:,n)));   % ���Ǹ����뵥Ԫ�ľ�ֵ
%     avepower=mean(abs(Data(:,n)).^2); % ���Ǹ���Ԫ����ֵ
%     scatter(1,n)=1-ave*ave/avepower; % ����ǡ��״����������ʽ��7.12������Ĺ�һ�����ȷ���
% end
% [DSV,DSL]=min(scatter);
% disp('�����Ե��һ������=');disp(DSV);
% if DSV <0.12
%     disp('�����Ե��һ������С��0.12');
% else
%     disp('�����Ե�Ч������');
% end
% % DSVΪ�����뵥Ԫ��һ�����ȷ������Сֵ��DSLΪ�õ�Ԫ���кţ������ǵ����Ե㷨����ȡ��һ�����ȷ���ֵ��С�ľ��뵥Ԫ���õ�Ԫ�ķ��ȱ仯�����С
% % DSV
% % ���������λ����
% for m=1:rows % m����ͬ�Ļز�����
%     RetData(m,:)=Data(m,:)*exp(-j*angle(Data(m,DSL))); % �����Ե㵥Ԫ����ĳһ�����뵥Ԫ������λ�������������뵥Ԫ����λ������Ӧ�þͿ��Բ��������࣬������ʱ���Ե㵥Ԫ�������ڵ���λ����0        
% end
% ang(1)=angle(Data(1,DSL)); % ang(1)�����Ե㵥Ԫ����ĳһ�����뵥Ԫ����1�����ڵ���λ
% for m=2:rows
%     ang(m)=ang(m-1)+angle(Data(m,DSL)*conj(Data(m-1,DSL))); % angʵ���Ͼ��ǵõ����Ե㵥Ԫ����ĳһ�����뵥Ԫ���ڲ�ͬ���ڵ���λ����m=2Ϊ����ang(2)=1������λ+��2������λ-1������λ��
%                                                           % ��ʵҲ�͵���2������λ
% end
% for m=1:rows
%     RetData(m,:)=Data(m,:)*exp(-j*ang(m));
% end

function RetData=Compensate(Data)
% %�����Ե㷨����У׼
[rows,cols]=size(Data);                   % ����cols�൱�ھ��뵥Ԫ������rows�൱�����ڸ�����Dara��ÿһ���ǰ�������ĸ�һά�����񣬲�ͬ���ж�Ӧ��ͬ������
Orth_Var=zeros(cols,1);                 
amp=abs(Data);                          % ����ȡ�˾���ֵ,amp��ÿһ���ǰ��������ʵһά�����񣬲�ͬ���ж�Ӧ��ͬ������
VarLim=0.12;                           % �趨һ������ֵ
Ang=zeros(1,rows);                      % �����ۻ���λ�������
for n=1:cols
    ave=mean(amp(:,n));                % �����뵥Ԫ���ֵ�ľ�ֵ
    avepower=mean(amp(:,n).^2);        % �����뵥Ԫ���ֵ�ľ���ֵ
    Orth_Var(n,1)=1-ave*ave/avepower;  % �����뵥Ԫ�Է��ȷ����һ���������״����������7.12��
end
[Sort_Var,Loc]=sort(Orth_Var);         % ������Թ�һ����ķ��ȷ������򲢶�λ��Sort_Var�ǰ��������к�Ľ����Loc��Sort_Var��Ԫ����֮ǰ�����е�λ��
if Sort_Var(1)>VarLim                  % �������������еģ�����ֻҪ��һ��Ԫ�ش������ޣ������о��뵥Ԫ�Ĺ�һ���������������0.12
    VarLim=Sort_Var(1);                % �����һ��Ԫ�ش������ޣ����޸�����ֵ
    disp('���о��뵥Ԫ��һ���������0.12��');
end
Len=0;
% ѡȡ���з��������ľ��뵥Ԫ
for n=1:cols
    if Sort_Var(n)<=VarLim
        Len=Len+1;
        DataOut(:,Len)=Data(:,Loc(n)); % �����������ľ��뵥Ԫ��������Ϊ���ز�����
        % Data��ĳһ�ж�Ӧĳһ�����뵥Ԫ�����ݣ�����ע��DataOut�еľ��뵥Ԫ�ǰ��շ��ȷ����һ������������е�
    end
end
% Len
SelVar=Sort_Var(1:Len);               % ѡȡ�������Ե�Ҫ��ľ��뵥Ԫ�ķ�ֵ��һ�������ΪSort_Var�Ǿ����������еĽ�������Է���Ҫ��Ŀ϶������е�ǰ��
Wei=(1./SelVar)/(sum(1./SelVar));     % ���ݷ������Ե�Ҫ��ķ�ֵ����������Ȩֵ
%%%%%%%%%%%%%%%%%%%% ������λ���Ȩ���ۣ�����ɻ��ۣ�
% for n=1:Len
%     Ang(1)=Ang(1)+Wei(n)*angle(xra(n,1));
% end
% for m=2:Col      % ��λ����������ѭ��
%     Ang(m)=Ang(m-1);
%     for n=1:Len  % ���Ե�ѭ��
%         Ang(m)=Ang(m)+Wei(n)*angle(DataOut(n,m)*conj(DataOut(n,m-1)));
%     end
% end
% for m=1:Col
%     xra(:,m)=xra(:,m).*exp(-j*Ang(m));
% end
%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%% �������ݼ�Ȩ�ڻ���
% ������������Ȩȡ�ڻ�
for n=1:Len
    Ang(1)=Ang(1)+Wei(n)*(DataOut(1,n)); % ���ȶԷ���Ҫ������о��뵥Ԫ�ĵ�1�����ڵ���������Ȩ,DataOut��ÿһ�ж�Ӧһ�����ڵ�����
end
Ang(1)=angle(Ang(1)); % ���Ǽ�Ȩ��õ��ĵ�һ�����ڵ���λ
% ��ʵ��������˵�ͺ͵����Ե����PhaseCompScaOne�еڶ��ִ����������ˣ�ֻ���������ĳ�λز���������������Ҫ��ľ��뵥Ԫͨ����Ȩ��͵õ���
for m=2:rows      % ��λ����������ѭ��
    for n=1:Len  % ���Ե�ѭ��
        Ang(m)=Ang(m)+Wei(n)*(DataOut(m,n)*conj(DataOut(m-1,n))); 
    end
    Ang(m)=Ang(m-1)+angle(Ang(m));
end   % ���յõ���Ang�е�Ԫ��Ϊ[1������λ 2������λ 3������λ ...]
for m=1:rows
    RetData(m,:)=Data(m,:).*exp(-j*Ang(m));
end