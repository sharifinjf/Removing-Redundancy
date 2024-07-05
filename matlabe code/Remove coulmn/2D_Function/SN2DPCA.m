function [ A ] = SN2DPCA(X,T,r,alpha,stop)
%Input:          X     -    m*n*s Image matrix. Each image have m*n pixels.
%                           s denote the number of images.
%                T     -    the number of iterations.
%                r     -    desired number of image's columns.
%                alpha -    specifies the positive ridge (L2) term
%                           coefficient.(Review SPCA)
%                stop  -    stop is the stopping criterion.(Review SPCA)
%Output:         A     -    n*r projection matrix.

%Example: Suppose Xtrain is traing sample and Xtest is testing sample. 
%         T=8; r=20; alpha=0.001;
%         stop=[];
%         for i=1:r
%             stop(i)=-randi([1 10]);
%         end
%         [ A ] = SN2DPCA(Xtrain,T,r,alpha,stop);
%         for i=1:size(Xtest,3)
%            Ytest(:,:,i)=Xtest(:,:,i)*A;     
%         end
        

%step1
%(Initialize B as arbitrary columnly-orthogonal n*r matrix. Initialize W1,W1,W2...Ws as m*m  identity matrix )
[m,n,s]=size(X);
B=eye(n,r);
W=zeros(m,m,s);
for i=1:s
    W(:,:,i)=eye(m,m);
end
 
%step2
for j=1:T
%   Given B update A.
    XWWX=zeros(n,n);
    for i=1:s
        XWWX=XWWX+X(:,:,i)'*W(:,:,i)'*W(:,:,i)*X(:,:,i);
    end
    [U1,D1,V1]=svd(XWWX);
    X1=D1^(1/2)*U1';
    for i=1:r
    A(:,i) = larsen(X1, X1*B(:,i), alpha, stop(i), [], false, false); 
    end
%   Given A update B
    XWWXA=XWWX*A;
    [Ud,D,V]=svd(XWWXA);
    U=Ud(1:n,1:r);
    B=U*V';
%   Update W
    for i=1:s
        W(:,:,i)=((X(:,:,i)-X(:,:,i)*A*B')*(X(:,:,i)-X(:,:,i)*A*B')')^(-1/4);
    end
end

%step3
%Normalize A
for l=1:r
    length=sqrt(sum(A(:,l).*A(:,l),1));     
    A(:,l)=A(:,l)./length;
end
    
end

