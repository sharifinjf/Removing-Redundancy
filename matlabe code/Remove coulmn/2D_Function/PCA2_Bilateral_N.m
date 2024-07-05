% written by Fanlong Zhang
% if you have any questions, please fell free to contact
% csfzhang@126.com

% version-20140701
% Copyright: PCA Lab, Nanjing University of Science and Technology, Nanjing

function [U,V,iter,obj]= PCA2_Bilateral_N(Train_DAT,proj_left,proj_right,tol)
% Image feature extraction using Bilateral N-2DPCA
% min sum(||Ai-UU'AiVV'||_*)
% Train_DAT:    Three dimension matrix: [Image_row, Image_column, Image_number]=size(Train_DAT);
% proj_left and proj_right: numbers of left and right features
% tol: terminal condition, default is 1e-3
if nargin < 4
   tol = 1e-3;
end
% initialization
[N,M,Train_NUM]=size(Train_DAT);
U=eye(N,proj_left);
maxIter =20;
converged = false;
iter = 1;
kesi=zeros(1,maxIter);
while ~converged
    iter=iter+1;
    %Compute right projection matrix V
   [V]=PCA2_N(Train_DAT,proj_right,U);
    %Compute left projection matrix U
    Train_DAT_new=zeros(M,N,Train_NUM);
    for k=1:Train_NUM
        Train_DAT_new(:,:,k)=Train_DAT(:,:,k)';  
    end    
   [U]=PCA2_N(Train_DAT_new,proj_left,V);
    % Evaluate the mean reconstruction error
    mre=kesi(iter);
    for k=1:Train_NUM
        S=Train_DAT(:,:,k);
        mre=mre+sum(svd(S-U*U'*S*(V*V')));  
    end
    kesi(iter)=mre/Train_NUM;
    obj=kesi*Train_NUM;
    % stop Criterion
    stopCriterion = abs((kesi(iter-1)-kesi(iter))/kesi(iter-1));
    if stopCriterion < tol
        converged = true;
    end     
    % disp(['Iteration' num2str(iter)  ' stopCriterion ' num2str(stopCriterion)]);    
    if ~converged && iter >= maxIter
      %  disp('Maximum iterations reached') ;
        converged = 1 ;       
    end   
end
end