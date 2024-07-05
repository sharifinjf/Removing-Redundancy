% written by Fanlong Zhang
% if you have any questions, please fell free to contact
% csfzhang@126.com

% version-20140701
% Copyright: PCA Lab, Nanjing University of Science and Technology, Nanjing

function [P]= PCA2_N(Train_DAT,r,Q)

[p,q,s]     = size(Train_DAT);
A           = cell(1,s);
for kk = 1:s
    A{kk} = Train_DAT(:,:,kk);
end

A_sum     = zeros(p,q);
for k=1:s
    A_sum = A_sum+A{k};
end

kesi      = ones(s,1);
%K=min(p,q)-1;
K          = 4;
gama       = 1;
iter_number= 3000;

QQ        = Q*Q';
D         = zeros(q,q);
%W=ones(p,p);
for k=1:s
    % D=D+A{k}'*W'*W*A{k};
    D     = D+A{k}'*(QQ-2*eye(size(QQ)))*QQ*A{k};
end

P_All          = cell(1,iter_number);
objective_iter = zeros(1,iter_number);

maxIter        = iter_number;
%tol=0.1;
tol            = 0.001;
converged      = false;
iter           = 0;
objective0     = 0;
while ~converged
    iter       = iter + 1;
    % Uptaing P
    [P]         = Find_Projection(D,r);
    P_All{iter} = P;
    
    %Updating W and D
    M           = P*P';
    Wk          = cell(s,1);
    D           = zeros(q,q);
    for k = 1:s
        AMk                 = A{k}-QQ*A{k}*M;
        [Ukesi,Skesi,Vkesi] = svd(AMk);
        kesi(k)             = min(kesi(k),gama*Skesi(K,K));
        middle              = max(diag(Skesi),kesi(k));
        Sk_kesi             = 1./(middle.^0.5);
        Wk{k}               = Ukesi(:,1: length(Sk_kesi))*diag(Sk_kesi)*Ukesi(:,1: length(Sk_kesi))';
        D                   = D+A{k}'*(QQ-2*eye(size(QQ)))*(Wk{k}'*Wk{k})*QQ*A{k};
    end
    
    %Evaluation
    objective              = 0;
    for k=1:s
        Au                 = A{k}-QQ*A{k}*(P*P');
        nuclear_k          = sum(svd(Au));
        objective          = nuclear_k+objective;
    end
    objective_iter(1,iter) = objective;
    
    % stop Criterion
    stopCriterion          = abs((objective0-objective)/objective0);
    if stopCriterion < tol
        converged          = true;
    end
    objective0=objective;
    %disp(['Iteration' num2str(iter)  ' stopCriterion ' num2str(stopCriterion)]);
    if ~converged && iter >= maxIter
        %disp('Maximum iterations reached') ;
        converged          = 1 ;
    end
end
end