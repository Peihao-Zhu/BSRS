function [x,beta,e] = channelpruning(A,b,lambda,alpha,max_iter)
%UNTITLED 此处显示有关此函数的摘要
%   此处显示详细说明
m = size(A,2);%200
e = zeros(max_iter,1);
x0 = rand(m,size(b,2))*0.01;

for j = 1:size(x0,1)
        x0(j,:) = x0(j,:)./norm(x0(j,:));
end
    
beta0 = ones(m,1);
for i = 1:max_iter
    i
    for j = 1:m
        z(:,:,j) = A(:,j)*x0(j,:);
    end
    vz = reshape(z,m,size(A,1)*size(b,2))';
    %res = lasso(vz,reshape(b,size(A,1)*size(b,2),1));
    vb = reshape(b,size(A,1)*size(b,2),1);
    for j = 1:1000
        g = vz'*vz*beta0-vz'*vb+lambda*sign(beta0);
        beta0 = beta0 -(alpha*(1000-j)/1000)*g;
    end
    %idx = 1+round(i/max_iter*lambda*size(res,2));
    %beta0 = res(:,ceil(0.5*size(res,2)));
    Abeta = A.*repmat(beta0',size(A,1),1);
    x0 = (Abeta'*Abeta)\Abeta'*b;
    for j = 1:size(x0,1)
        x0(j,:) = x0(j,:)./norm(x0(j,:));
    end
end

for j = 1:m
        z(:,:,j) = A(:,j)*x0(j,:);
end
vz = reshape(z,m,size(A,1)*size(b,2))';
%res = lasso(vz,reshape(b,size(A,1)*size(b,2),1));
vb = reshape(b,size(A,1)*size(b,2),1);
for j = 1:1000
    g = vz'*vz*beta0-vz'*vb+lambda*sign(beta0);
    beta0 = beta0 -(alpha*(1000-j)/1000)*g;
end

x = x0;
beta=beta0;
end

