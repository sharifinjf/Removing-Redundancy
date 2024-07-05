%%                 IN THE NAME OF ALLAH
%[1] W.Zuo,D.Zhang,and K.Wang,“Bidirectional PCA with assembled matrix distance metric for image recognition,
%”IEEE Trans.Syst.Man,Cybern.Part B Cybern.,vol.36,no.4,pp.863–872,2006.
function [w_col,w_row]=B2DPCA(x)
[m,n,N]         = size(x);                   % size of observation 
x               = double(x);                
mu              = mean(x,3);                 % calculate mean af image 
x               = x-repmat(mu,[1 1 N]);      % centering each  observation
s_r1            = zeros([n n N]);
s_c1            = zeros([m m N]);
for i=1:N
    s_r1(:,:,i) = x(:,:,i)'*x(:,:,i);         % alculate  row covariance for each observation
    s_c1(:,:,i) = x(:,:,i)*x(:,:,i)';         % alculate  column covariance for each observation
end
s_row           = N/m*mean(s_r1,3);           % sum row covariance matrix 
s_col           = N/n*mean(s_c1,3);           % sum column covariance matrix
[u_row,l_row]   = eig(s_row);                 %compute eigenvalues and eigenvectors
[u_col,l_col]   = eig(s_col); 
[~,index_row]   = sort(abs(diag(l_row)),'descend'); % find max eigenvalues
[~,index_col]   = sort(abs(diag(l_col)),'descend');
w_row           = u_row(:,index_row);               % select eigenvector crospond to maximom eigenvalus
w_col           = u_col(:,index_col);
end
%%
% % dr=40;
% % dc=50;
% % test=w_col(:,1:dc)'*double(orl_database1(:,:,1))*w_row(:,1:dr);
% % r_test=w_col(:,1:dc)*test*w_row(:,1:dr)';
% % imshow(r_test,[0 255]);