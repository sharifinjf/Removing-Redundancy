 
% LLE algorithm(using K nearest neighbors)
% [Y] = lle(X,K,dmax)
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

X2 = sum(X.^2,1);
distance = repmat(X2,N,1)+repmat(X2',1,N)-2*X'*X;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[sorted,index] = sort(distance);%[B,ind]=sort(A)£¬
neighborhood = index(2:(1+K),:);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% STEP2: SOLVE FOR RECONSTRUCTION WEIGHTS
fprintf(1,'-->Solving for reconstruction weights.\n');

if(K>D) 
  fprintf(1,'   [note: K>D; regularization will be used]\n'); 
  tol=1e-2; % regularlizer in case constrained fits are ill conditioned
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


%