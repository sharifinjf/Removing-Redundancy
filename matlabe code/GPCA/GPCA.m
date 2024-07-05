%%           IN THE NAIME AF ALLAH
% [1] Jieping Ye, Ravi Janardan, and Qi Li. 2004. GPCA: an efficient dimension reduction scheme for image compression and retrieval.
% In Proceedings of the tenth ACM SIGKDD international conference on Knowledge discovery and data mining (KDD '04).
% ACM, New York, NY, USA, 354-363. DOI=http://dx.doi.org/10.1145/1014052.1014092
%
% Written by: Ali Asghar Sharifi
% Last update: 2017/04/22
%%
clear variables; clc; close all;
%% initialization
Max_itr  = 10;  % Maximum Iteration Number
d_r      = 10;  % Right Dimension
d_l      = 20;  % Left Dimension
%% load data
load('F:\thesis\database\img-database\orl_faces\orl_database1.mat')
x          = double(orl_database1);    %orginal data
% load('F:\thesis\database\yalefaces\yale\Yeal_database.mat')
% x         = Yeal;
%% PIE database
% load('F:\thesis\database\img-database\PIE\PIE_32x32.mat')
% X           = fea
%% centering for each image
% % [n,m,p]    =  size(x);
% % x_re       =  reshape(x,[],p);
% % mu_img     =  mean(x_re);
% % x_re_ce    =  x_re-repmat(mu_img,n*m,1);
% % x          = reshape(x_re_ce,n,m,p);
%% centering for each pixel image
[n,m,p]   = size(x);
mu        = mean(x,3);                 % mean data
x_cen     = x - repmat(mu,1,1,p);      % centering
%% GPCA algorithm

L      = [eye(d_l) zeros(d_l,n-d_l)]';     % intializiton for Left transform matrix
RMSE   = zeros(Max_itr,1);
for i= 1:Max_itr
    M_R    = zeros(m,m);                    % Right Covariance Matrix
    for jj = 1:p
        M_R =  M_R + x_cen(:,:,jj)'*(L*L')*x_cen(:,:,jj);
    end
    [D_r,r]           = eig(M_R);                         % compute eigenvalue matirx M_L
    [value_r,index_r] = sort((max(((r)))),'descend');     % sort eigenvalue
    R                 = (D_r(:,index_r(1:d_r)));          % select d_l eigenvector crosponding to largest eigenvalue
    
    M_L    = zeros(n,n);  % Left Covariance Matrix
    for nn = 1:p
        M_L = M_L + x_cen(:,:,jj)*(R*R')*x_cen(:,:,jj)';
    end
    [D_l,l]           = eig(M_L);                          % compute eigenvalue matirx M_L
    [value_l,index_l] = sort((max(((l)))),'descend');      % sort eigenvalue
    L                 = (D_l(:,index_l(1:d_l)));           % select d_l eigenvector crosponding to largest eigenvalue
    
    x_hat_ce = zeros(n,m,p);
    for mm = 1:p
        x_hat_ce(:,:,mm) = (L*L')*x_cen(:,:,mm)*(R*R');
    end
    error = ((x_hat_ce - x_cen).^2);                      % compute Frobenius norm for error reconstrection
    RMSE(i,1) = sqrt((1/p)*sum(error(:)));                % copmute RMSE
    %     clear jj nn mm;
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% B2DPCA
 [w_col,w_row] = B2DPCA(x);
 y             = w_col(:,1:d_l)'*x(:,:,1)*w_row(:,1:d_r);
 x_hat         = w_col(:,1:d_l)*y*w_row(:,1:d_r)';
 figure(1)
 imshow (x_hat,[0 255])
%%
figure(2)
plot (1:Max_itr ,RMSE,'-*')
ylabel('RMSE')
xlabel('NUMBER ITERATION')
legend('ORL database')
figure (3)
x_hat = x_hat_ce + repmat(mu,1,1,p);
% % % if (min(x_hat(:))<0)
% % %     x_hat = x_hat - min(x_hat);
% % %     x_hat = (x_hat*max(x(:)))/max(x_hat(:));
% % % end
% % %
imshow(x_hat (:,:,2),[0 255])
%%
