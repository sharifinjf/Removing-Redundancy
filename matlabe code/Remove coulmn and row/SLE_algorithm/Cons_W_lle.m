%
%2009.04.11 黄传波
% LLE的程序
% LLE algorithm(算法) (using K nearest neighbors)
%
% [Y] = lle(X,K,dmax)
%
% X = data as D x N matrix (D = dimensionality, N = #points)
% K = number of neighbors
% dmax = max embedding dimensionality
% Y = embedding as dmax x N matrix

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [W2] = Cons_W_lle(X,K)
%
[D,N] = size(X);
fprintf(1,'LLE running on %d points in %d dimensions\n',N,D);
if K>=N
    K=N-2;
end

% STEP1: COMPUTE PAIRWISE DISTANCES & FIND NEIGHBORS 
fprintf(1,'-->Finding %d nearest neighbours.\n',K);
%计算点对距离（欧氏距离）
X2 = sum(X.^2,1);%矩阵X每个元素平方，再每列相加，而得到的一个行向量
distance = repmat(X2,N,1)+repmat(X2',1,N)-2*X'*X;%B = repmat(A,m,n)将矩阵A复制m×n块，即B由m×n块A平铺而成。
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%找K个近邻点
[sorted,index] = sort(distance);%[B,ind]=sort(A)，计算后，B是A排序后的向量，ind是B中每一项对应于A中项的索引,排序是安升序进行的
neighborhood = index(2:(1+K),:);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% STEP2: SOLVE FOR RECONSTRUCTION WEIGHTS
fprintf(1,'-->Solving for reconstruction weights.\n');

if(K>D) 
  fprintf(1,'   [note: K>D; regularization will be used]\n'); 
  tol=1e-2; % regularlizer in case constrained(限制) fits are ill conditioned
else
  tol=0.1;
end

W = zeros(K,N);
for ii=1:N
   z = X(:,neighborhood(:,ii))-repmat(X(:,ii),1,K); % shift ith pt to origin
   C = z'*z;                                        % local covariance
   C = C + eye(K,K)*tol*trace(C);                   % regularlization (K>D)
   W(:,ii) = C\ones(K,1);                           % solve Cw=1
   %W(:,ii) = W(:,ii)/sum(W(:,ii));                  % enforce sum(w)=1
   a=sum(W(:,ii));
   if a==0
       a=1;
   end
   W(:,ii) = W(:,ii)/a;
end;
W2=zeros(N,N);
for i=1:N
   W2(i,neighborhood(:,i)')=W(:,i)';
end

% STEP 3: COMPUTE EMBEDDING FROM EIGENVECTS OF COST MATRIX M=(I-W)'(I-W)
fprintf(1,'-->Computing embedding.\n');

% M=eye(N,N); % use a sparse matrix with storage for 4KN nonzero elements
% M = sparse(1:N,1:N,ones(1,N),N,N,4*K*N); 
% for ii=1:N
%    w = W(:,ii);
%    jj = neighborhood(:,ii);
%    M(ii,jj) = M(ii,jj) - w';
%    M(jj,ii) = M(jj,ii) - w;
%    M(jj,jj) = M(jj,jj) + w*w';
% end;

% CALCULATION OF EMBEDDING
%options.disp = 0; options.isreal = 1; options.issym = 1; 
%[Y,eigenvals] = eigs(M,d+1,0,options);
%Y = Y(:,2:d+1)'*sqrt(N); % bottom evect is [1,1,1,1...] with eval 0


%fprintf(1,'Done.\n');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% other possible regularizers for K>D
%   C = C + tol*diag(diag(C));                       % regularlization
%   C = C + eye(K,K)*tol*trace(C)*K;                 % regularlization