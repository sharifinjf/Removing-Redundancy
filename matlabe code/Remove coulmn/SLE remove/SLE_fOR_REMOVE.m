%%                              IN THE NAIME OF ALLAH
% Sparse linear embedding framework (similar to SPCA)
% note : preprocessing norm(x)=1 for any x may be very important to increase the performance!!
% this programme is used in the paper:
% Zhihui Lai, Waikeung Wong, Yong Xu, Jian Yang, Jinhui Tang and David
% Zhang, Approximate orthogonal sparse embedding for dimensionality reduction, IEEE Transactions on Neural Networks and Learning Systems, 2016, 27(4): 723-735.

function[eigvector,Xnew ] = SLE_fOR_REMOVE(X,Kneighbor,stop,lambda,max_dimension)

%% pre-dimensionality reduction whit SPCA
% %   [eigvector, ~, ~]  = PCA_dencai(X',10);                  %input row vector is a sample image, for pre-dimensionality reduction to accelerate the speed in SPCA
% %   projection          = eigvector;%(:,1:dim);               % eigen vector PCA
% %   Ytrain             = projection'*X;                      % new matrix train
%% LLE and calculate Graph matrix
Ytrain             = X;
Xforconstructgraph = Ytrain;
%construct the LLE matrix as the same one of the Science paper
W                  = Cons_W_lle(Xforconstructgraph,Kneighbor);
%------------constructing within-class reconstruction matrix__________
%construct the matrix as I-W
ImW                = eye(size(W))-W;

%construct the special scatter matric for SLE used for SPCA regression (it is the equivalent representation of the formula in the paper)
% XX               = Xforconstructgraph*ImW'*ImW*Xforconstructgraph';
Xnew               = ImW*Xforconstructgraph';   
maxiter            = 50;
trace              = 1;
[D,N] = size(Xnew );
fprintf(1,'SPCA running on %d points in %d dimensions\n',N,D);
[sl,~,~,~,~,~]     = spca(Xnew  ,[], max_dimension, lambda, stop, maxiter, trace);%use the equivalent representation of eq.(19-20) so that it is not necessary to rewrite the key parts of the codes)
sl                 = fliplr(sl);                      % for the stability and obtaining the sparse projection of SLE
eigvector          = [sl ones(size(sl,1),1)];         % just a trick in dimensionality reduction to achieve the expected number of projections (for compensation)

% %            for dim = 1:max_dimension
% %             projection             = eigvector(:,1:dim);              % select ptojection vectore
% %             YYtrain                = projection'*Ytrain;              % fainal traint sequnce
% %             YYtest                 = projection'*Ytest;               % fainal test sequnce
% %             re_rat_SLE_KNN(Kneighbor,stop,dim)= KNN_Classfier(YYtrain', lable_train, YYtest',lable_test, 1); %Kneighbor,stop,
% %            end


end

