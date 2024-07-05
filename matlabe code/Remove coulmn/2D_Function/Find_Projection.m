% written by Fanlong Zhang
% if you have any questions, please fell free to contact
% csfzhang@126.com

% version-20140701



function [P]=Find_Projection(D,r)
addpath FOptM-share;

%n = size(D,1);X0 = randn(n,r);X0 = orth(X0);
[X0]=Eigenface_f((D+D')/2,r);
%[X0]=Find_K_Max_Eigen((D+D')/2,r);
%[X0]=Find_K_Max_Eigen(E,r);

opts.record = 0; %
opts.mxitr  = 1000;
opts.xtol = 1e-5;
opts.gtol = 1e-5;
opts.ftol = 1e-8;

[P]= OptStiefelGBB(X0, @fun2, opts, D);

end

function [disc_set,disc_value,Mean_Image]=Eigenface_f(Train_SET,Eigen_NUM)

% the magnitude of eigenvalues of this function is corrected right !!!!!!!!!
% Centralized PCA
[NN,Train_NUM]=size(Train_SET);

if NN<=Train_NUM % for small sample size case    
   Mean_Image=mean(Train_SET,2);  
   Train_SET=Train_SET-Mean_Image*ones(1,Train_NUM);
   R=Train_SET*Train_SET'/(Train_NUM-1);
   
   [V,S]=Find_K_Max_Eigen(R,Eigen_NUM);
   disc_value=S;
   disc_set=V;

else % for small sample size case
    Mean_Image=mean(Train_SET,2);  
   Train_SET=Train_SET-Mean_Image*ones(1,Train_NUM);
   R=Train_SET'*Train_SET/(Train_NUM-1);
  
  [V,S]=Find_K_Max_Eigen(R,Eigen_NUM);
  disc_value=S;
  disc_set=zeros(NN,Eigen_NUM);
  
  Train_SET=Train_SET/sqrt(Train_NUM-1);
  for k=1:Eigen_NUM
    disc_set(:,k)=(1/sqrt(disc_value(k)))*Train_SET*V(:,k);
  end
end
end