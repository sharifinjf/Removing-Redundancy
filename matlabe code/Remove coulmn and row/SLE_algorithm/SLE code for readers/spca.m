function [sl sv pcal pcav paths] = spca(X, Gram, K, lambda, stop, maxiter, trace)
% SPCA  The SPCA algorithm for computing sparse principal components.
%    [SL SV PCAL PCAV PATHS] = SPCA(X, Gram, K, LAMBDA, STOP) computes
%    sparse principal components of the data in X. X is an n x p matrix
%    where n is the number of observations and p is the number of
%    variables. Gram = X'X is the p x p Gram matrix. Either X, Gram or
%    both may be supplied. Pass an empty matrix as argument if either of X
%    or Gram is missing.  
%
%    Returns SL, the sparse loadings (aka principal component directions),
%    SV, the variances of each component, PCAL, the loadings of a regular
%    PCA, PCAV, the variances of a regular PCA and PATHS, an optional
%    struct containing the loading paths for each component as functions of
%    iteration number.
%
%    K is the desired number of sparse principal components. 
%
%    LAMBDA specifies the positive ridge term coefficient. If LAMBDA is set
%    to infinity, soft thresholding is used to calculate the components.
%    This is appropriate when p>>n and results in a significantly faster
%    algorithm.
%
%    STOP is the stopping criterion. If STOP is negative, its absolute
%    value corresponds to the desired number of variables. If STOP is
%    positive, it corresponds to an upper bound on the L1-norm of the BETA
%    coefficients. STOP = 0 results in a regular PCA. Negative STOP cannot
%    be used with LAMBDA set to infinity. 
%
%    The extra inputs MAXITER and TRACE determine the maximum number of
%    iterations and control output. 
%
%    Note that if X is omitted, the absolute values of SV cannot be
%    trusted, however, the relative values will still be correct.
%
% Author: Karl Skoglund, IMM, DTU, kas@imm.dtu.dk
% Reference: 'Sparse Principal Component Analysis' by Hui Zou, Trevor
% Hastie and Robert Tibshirani, 2004.

%% Input checking and initialization
if nargin < 7
  trace = 0;
end
if nargin < 6
  maxiter = 300;
end
if nargin < 5
  stop = 0;
end
if nargin < 4
  lambda = 1e-6;
end
if nargin < 3
  error('Minimum three arguments are required');
end
if nargout == 5
  savepaths = 1;
else
  savepaths = 0;
end
if isempty(X) && isempty(Gram)
  error('Must supply a data matrix or a Gram matrix or both.');
end
if lambda == inf && any(stop < 0)
  warning('Cannot use negative STOP (number of variables criterion) with LAMBDA = inf (soft thresholding).');
end

%% SPCA algorithm setup

if isempty(X)%X为空集则执行对Gram矩阵进行分解,通常X肯定存在的
  [V D] = eig(Gram);
  X = V*sqrt(abs(D))*V';
end
[U D pcal] = svd(X, 'econ');
[n p] = size(X);
pcav = diag(D).^2/n;

K = min([K p n-1]);

if savepaths
  for k = 1:K
    paths(k).data = [];
  end
end

A = pcal(:,1:K);
B_norm = zeros(p,K);
iter = 0;
converged = 0;

%% SPCA loop
while ~converged && iter < maxiter
  iter = iter + 1;
  
  if trace && ~mod(iter, 10)
    disp(['Iteration ' num2str(iter) ', diff = ' num2str(max(abs(B_old(:) - B_norm(:))))]);
  end
  B_old = B_norm;

  for j = 1:K
    if length(stop) == K
      jstop = stop(j);
    else
      jstop = stop(1);
    end
    if lambda == inf
      % Soft thresholding, calculate beta directly
      if isempty(Gram)
        AXX = (A(:,j)'*X')*X;
      else
        AXX = A(:,j)'*Gram;
      end
      b = sign(AXX).*max(0, abs(AXX) - jstop);
    else
      % Find beta by elastic net regression
      b = larsen(X, X*A(:,j), lambda, jstop, 0);%调用了另一个function!!!!!!!!!!!!!!!!!!
    end
    B(:,j) = b(end,:)';%量后一行的转置？？
  end
obj(iter)=norm(B_old - B);%max(abs(B_old(:) - B_norm(:)));%
  % Normalize coefficients
  B_norm = sum(B.^2);
  B_norm(B_norm == 0) = 1;
  B_norm = B./sqrt(ones(p,1)*B_norm);

  converged = max(abs(B_old(:) - B_norm(:))) < 1e-6;
  
  % Save coefficient data
  if savepaths
    for k = 1:K
      paths(k).data = [paths(k).data B(:,k)];
    end
  end

  % Update A
  if isempty(Gram)
    [U D V] = svd(X'*(X*B), 'econ');
  else
    [U D V] = svd(Gram*B, 'econ');
  end    
  A = U*V';
end

%% Normalization of loadings
% Normalize coefficients such that loadings has Euclidian length 1
B_norm = sum(B.^2);
B_norm(B_norm == 0) = 1;
sl = B./sqrt(ones(p,1)*B_norm);

%% Order modes such that maximal total explained variance is achieved
ss = X*sl; % sparse scores
sv = zeros(K, 1); % adjusted variances
O = 1:K; % ordering
for k = 1:K
  ss_var = sum(ss.^2)/n; % variances of scores
  [sv(k) max_col] = max(ss_var);
  s = ss(:,max_col); % column to factor out
  s_norm = s'*s;
  if s_norm > eps,
    O(O == max_col) = O(k);
    O(k) = max_col;
    ss(:,O) = ss(:,O) - s*s'*ss(:,O)/s_norm; % factor out chosen column
  end
end
sl = sl(:,O); % change order of loadings

%% Print information
if trace
  if p < 20
    disp(sprintf('\n\n --- Sparse loadings ---\n'));
    disp(sl)
  end
  disp(sprintf('\n --- Adjusted variances, Variance of regular PCA ---\n'));
  disp([sv/sum(pcav) pcav(1:K)/sum(pcav)])
  disp(sprintf('Total: %3.2f%% %3.2f%%', 100*sum(sv/sum(pcav)), 100*sum(pcav(1:K)/sum(pcav))));
  disp(sprintf('\nNumber of nonzero loadings:'));
  disp(sum(abs(sl) > 0));  
end
