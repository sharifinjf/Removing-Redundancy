%%                   IN THE NAME AF ALLAH
% [1]J.Yang,D.Zhang,A.F.Frangi,andJ.Y.Yang,“Two-Dimensional PCA: A New Approach to Appearance-Based Face Representation and Recognition,”...
%IEEE Trans.Pattern Anal.Mach.Intell.,vol.26,no.1,pp.131–137,2004.

function [w]     = D2PCA(x)
[m,n,M]          = size(x);                      % size of observation 
x                = double(x);                
mu               = mean(x,3);                    % calculate mean af image 
x                = x-repmat(mu,[1 1 M]);         % centering each  observation
g                = zeros([n n M]);
for i=1:M
    g(:,:,i)     = x(:,:,i)'*x(:,:,i);           % calculate  row covariance for each observation
end
g_t              = mean(g,3);                    % sum row covariance matrix 
[u,l]            = eig(g_t);                     %compute eigenvalues
[~,index]        = sort(abs(diag(l)),'descend'); % find max eigenvalues
w                = u(:,index);                   %select eigenvector crospond to maximom eigenvalus
end