function [out,E] = FMEPC(sig)
%FMEPC 此处显示有关此函数的摘要
%   此处显示详细说明
    M=size(sig,1);
    N=size(sig,2);
    G=fft(sig,[],1);
    theta=(rand(M,1)-0.5)*2*zeros(1,N);
    comp=exp(-1i*theta);
    for i=1:10000
        G=G.*comp;
        I=fft(G,[],1);
%         I=fft(G.*comp,[],1);
        
        E(i)=myE(I);
        if i>1 && abs(E(i-1)-E(i))<1e-9
            out=I;
            break;
        end
%         w=sum(G.*(fft(log(abs(I)).*conj(I),[],1)),2);
        w=sum(G.*(fft(log(abs(I)).*conj(I),[],1)),2);
        comp=(conj(w)./abs(w))*ones(1,N);
    end
    out=I;
end


function [out]=myE(sig)
    D=(abs(sig)).^2/sum(sum((abs(sig)).^2));
    out=-sum(sum(D.*log(D)));
end

