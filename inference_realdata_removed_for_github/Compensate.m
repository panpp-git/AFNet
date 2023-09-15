% function RetData=Compensate(Data)
% %互相关相位对准
% [rows,cols]=size(Data);
% xw1=zeros(1,cols);
% xw2=zeros(1,cols);
% RetData(1,:)=Data(1,:);
% for k=2:rows
%     xw1=Data(k,:).*conj(RetData(k-1,:));%相邻两次回波共轭相乘
%     xw2=abs(xw1);%
%     sum1=sum(xw1);%
%     sum2=sum(xw2);%
%     RetData(k,:)=Data(k,:)*conj(sum1/sum2);
% end

% function RetData=Compensate(Data)
% %单特显点法初相校准
% [rows,cols]=size(Data);%行：方位 列：距离
% scatter=zeros(1,cols);
% for n=1:cols
%     ave=mean(abs(Data(:,n)));   % 这是各距离单元的均值
%     avepower=mean(abs(Data(:,n)).^2); % 这是各单元均方值
%     scatter(1,n)=1-ave*ave/avepower; % 这就是《雷达成像技术》公式（7.12）定义的归一化幅度方差
% end
% [DSV,DSL]=min(scatter);
% disp('单特显点归一化幅度=');disp(DSV);
% if DSV <0.12
%     disp('单特显点归一化幅度小于0.12');
% else
%     disp('单特显点效果不佳');
% end
% % DSV为各距离单元归一化幅度方差的最小值，DSL为该单元序列号，由于是单特显点法，就取归一化幅度方差值最小的距离单元，该单元的幅度变化起伏最小
% % DSV
% % 下面进行相位补偿
% for m=1:rows % m代表不同的回波周期
%     RetData(m,:)=Data(m,:)*exp(-j*angle(Data(m,DSL))); % 用特显点单元（即某一个距离单元）的相位来补偿各个距离单元的相位，这样应该就可以补偿掉初相，并且这时特显点单元各个周期的相位都是0        
% end
% ang(1)=angle(Data(1,DSL)); % ang(1)是特显点单元（即某一个距离单元）第1个周期的相位
% for m=2:rows
%     ang(m)=ang(m-1)+angle(Data(m,DSL)*conj(Data(m-1,DSL))); % ang实际上就是得到特显点单元（即某一个距离单元）在不同周期的相位。以m=2为例，ang(2)=1周期相位+（2周期相位-1周期相位）
%                                                           % 其实也就等于2周期相位
% end
% for m=1:rows
%     RetData(m,:)=Data(m,:)*exp(-j*ang(m));
% end

function RetData=Compensate(Data)
% %多特显点法初相校准
[rows,cols]=size(Data);                   % 这里cols相当于距离单元个数，rows相当于周期个数，Dara的每一行是包络对齐后的复一维距离像，不同的列对应不同的周期
Orth_Var=zeros(cols,1);                 
amp=abs(Data);                          % 由于取了绝对值,amp的每一行是包络对齐后的实一维距离像，不同的行对应不同的周期
VarLim=0.12;                           % 设定一个门限值
Ang=zeros(1,rows);                      % 用以累积相位差的向量
for n=1:cols
    ave=mean(amp(:,n));                % 按距离单元求幅值的均值
    avepower=mean(amp(:,n).^2);        % 按距离单元求幅值的均方值
    Orth_Var(n,1)=1-ave*ave/avepower;  % 按距离单元对幅度方差归一化，见《雷达成像技术》（7.12）
end
[Sort_Var,Loc]=sort(Orth_Var);         % 按升序对归一化后的幅度方差排序并定位，Sort_Var是按升序排列后的结果，Loc是Sort_Var中元素在之前序列中的位置
if Sort_Var(1)>VarLim                  % 由于是升序排列的，所以只要第一个元素大于门限，则所有距离单元的归一化方差均大于门限0.12
    VarLim=Sort_Var(1);                % 如果第一个元素大于门限，则修改门限值
    disp('所有距离单元归一化方差大于0.12！');
end
Len=0;
% 选取所有符合条件的距离单元
for n=1:cols
    if Sort_Var(n)<=VarLim
        Len=Len+1;
        DataOut(:,Len)=Data(:,Loc(n)); % 将符合条件的距离单元重新排列为复回波矩阵
        % Data中某一列对应某一个距离单元的数据，但是注意DataOut中的距离单元是按照幅度方差归一化结果升序排列的
    end
end
% Len
SelVar=Sort_Var(1:Len);               % 选取符合特显点要求的距离单元的幅值归一化方差，因为Sort_Var是经过升序排列的结果，所以符合要求的肯定在序列的前面
Wei=(1./SelVar)/(sum(1./SelVar));     % 根据符合特显点要求的幅值方差矩阵计算权值
%%%%%%%%%%%%%%%%%%%% 相邻相位差加权积累（非相干积累）
% for n=1:Len
%     Ang(1)=Ang(1)+Wei(n)*angle(xra(n,1));
% end
% for m=2:Col      % 方位采样脉冲数循环
%     Ang(m)=Ang(m-1);
%     for n=1:Len  % 特显点循环
%         Ang(m)=Ang(m)+Wei(n)*angle(DataOut(n,m)*conj(DataOut(n,m-1)));
%     end
% end
% for m=1:Col
%     xra(:,m)=xra(:,m).*exp(-j*Ang(m));
% end
%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%% 相邻数据加权内积法
% 对相邻向量加权取内积
for n=1:Len
    Ang(1)=Ang(1)+Wei(n)*(DataOut(1,n)); % 首先对符合要求的所有距离单元的第1个周期的数据做加权,DataOut中每一列对应一个周期的数据
end
Ang(1)=angle(Ang(1)); % 这是加权后得到的第一个周期的相位
% 其实到这里来说就和单特显点程序PhaseCompScaOne中第二种处理方法类似了，只不过这里的某次回波数据是所有满足要求的距离单元通过加权求和得到的
for m=2:rows      % 方位采样脉冲数循环
    for n=1:Len  % 特显点循环
        Ang(m)=Ang(m)+Wei(n)*(DataOut(m,n)*conj(DataOut(m-1,n))); 
    end
    Ang(m)=Ang(m-1)+angle(Ang(m));
end   % 最终得到的Ang中的元素为[1周期相位 2周期相位 3周期相位 ...]
for m=1:rows
    RetData(m,:)=Data(m,:).*exp(-j*Ang(m));
end