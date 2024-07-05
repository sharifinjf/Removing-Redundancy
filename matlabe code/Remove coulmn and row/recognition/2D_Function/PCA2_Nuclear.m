% written by Fanlong Zhang
% if you have any questions, please fell free to contact
% csfzhang@126.com

% version-20140701
% Copyright: PCA Lab, Nanjing University of Science and Technology, Nanjing


function [U,Wk,objective_iter,U_All] = PCA2_Nuclear(Train_DAT,r,tol)

% Image feature extraction using  N-2DPCA
% min sum(||Ai-UU'Ai||_*)
% Train_DAT:    Three dimension matrix: [Image_row, Image_column, Image_number]=size(Train_DAT);
% r: numbers of feature
% tol: terminal condition, default is 1e-6
if nargin < 3
    tol         = 1e-6;
end

[p,q,s]         = size(Train_DAT);
A               = cell(1,s);
for kk = 1:s
    A{kk}       = Train_DAT(:,:,kk);
end

A_sum           = zeros(p,q);
for k = 1:s
    A_sum       = A_sum+A{k};
end

kesi            = ones(s,1);
%K=min(p,q)-1;
K               = 4;
gama            = 1;
iter_number     = 3000;
D               = zeros(q,q);
W               = eye(p,p);
for k=1:s
    D           = D+A{k}'*W'*W*A{k};
    %D=D+A{k}'*A{k};
end

U_All          = cell(1,iter_number+1);
U_All{1}       = 0;
objective_iter = zeros(1,iter_number);

maxIter        = iter_number;
%tol=0.1;
converged        = false;
iter             = 0;
objective0       = 0;
while ~converged
    iter         = iter + 1;
    % Uptaing U
    [U]          = Find_K_Max_Eigen(D,r);
    U_All{iter+1}= U;
    
    %Updating W and D
    M            = U*U';
    D_pre        = D;
    
    CkCk         = cell(s,1);
    Wk           = cell(s,1);
    
    for k = 1:s
        AMk                = A{k}*(eye(q)-M); %disp(rank(AMk));
        [Ukesi,Skesi,Vkesi]= svd(AMk);
        kesi(k)            = min(kesi(k),gama*Skesi(K,K));
        
        middle             = max(diag(Skesi),kesi(k));
        Sk_kesi            = 1./(middle.^0.5);
        
        Wk{k,1}            = Ukesi(:,1: length(Sk_kesi))*diag(Sk_kesi)*Ukesi(:,1: length(Sk_kesi))';
        
        Ck                 = Wk{k,1}*A{k};
        CkCk{k,1}          = Ck'*Ck;
    end
    %disp(disg(Skesi));
    D                      = zeros(q,q);
    for k = 1:s
        D                  = CkCk{k,1}+D;
    end
    
    %disp([norm(D-D_pre,'fro') norm(U_All{iter+1}-U_All{iter},'fro')]);
    %Evaluation
    objective              = 0;
    for k = 1:s
        Au                 = A{k}-A{k}*U*U';
        [Ua,Sa,Va]         = svd(Au);
        nuclear_k          = sum(sum(diag(Sa)));
        objective          = nuclear_k+objective;
    end
    objective_iter(1,iter) = objective;
    
    % stop Criterion
    stopCriterion          = abs(objective0-objective);
    if stopCriterion < tol
        converged          = true;
    end
    objective0=objective;
    %disp(['Iteration' num2str(iter)  ' stopCriterion ' num2str(stopCriterion)]);
    if ~converged && iter >= maxIter
        %disp('Maximum iterations reached') ;
        converged = 1 ;
    end
end