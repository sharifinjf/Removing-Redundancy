%%                              IN THE NAIME OF ALLAH
% Sparse linear embedding framework (similar to SPCA)
% note : preprocessing norm(x)=1 for any x may be very important to increase the performance!!
% this programme is used in the paper:
% Zhihui Lai, Waikeung Wong, Yong Xu, Jian Yang, Jinhui Tang and David
% Zhang, Approximate orthogonal sparse embedding for dimensionality reduction, IEEE Transactions on Neural Networks and Learning Systems, 2016, 27(4): 723-735.

function[re_rat_SLE_KNN] = SLE(x_train,lable_train,lable_test,x_test,max_dimension,numbertrainingsample,numberclass)
k = 0;
trainingnuminclass = numbertrainingsample;
classnum           = numberclass;
[~,~,p]            = size(x_train);
[~,~,p1]           = size(x_test);
%% train and test sequnce
Xtrain             = reshape(x_train,[],p);           % convert image to vectore
Xtest              = reshape(x_test,[],p1);           % convert image to vectore
%% pre-dimensionality reduction whit SPCA
[eigvector, ~, ~] = PCA_dencai(Xtrain', 150); %input row vector is a sample image, for pre-dimensionality reduction to accelerate the speed in SPCA
projection = eigvector;%(:,1:dim);                    % eigen vector PCA
Ytrain     = projection'*Xtrain;                      % new matrix train
Ytest      = projection'*Xtest;%                      % new matrix test
%% LLE and calculate Graph matrix
Xforconstructgraph = Ytrain;
KK                 = [3 9 27 trainingnuminclass*classnum-2];
for Kneighbor = 1:4%:5%:100%2:2:20
    %construct the LLE matrix as the same one of the Science paper
    W       = Cons_W_lle(Xforconstructgraph,KK(Kneighbor));   
    %------------constructing within-class reconstruction matrix__________
    %construct the matrix as I-W
    ImW     = eye(size(W))-W;    
    %construct the special scatter matric for SLE used for SPCA regression (it is the equivalent representation of the formula in the paper)
    XX      = Xforconstructgraph*ImW'*ImW*Xforconstructgraph';
    %--------------------------------SPCA preparition------------------------
    %                         XX=(Xforconstructgraph*ImW')';clear projectmat
    %                         XX = normalize(XX);%X is :each row is a normalized vector.
    %                         [n p] = size(XX);
    %                         K = size(XX,2);%6
    %you can set the following parameters
    lambda  = 1000;
    maxiter = 50;
    trace   = 1;
    %% SLE algorithm
    % ---------------------------------perform SPCA----------------------------
    for stop = 1:3  %  size(XX,2)-1%stop=36:92.5%
        Nprojection                = 150;
        [sl,~,~,~,~,~] = spca([], XX, Nprojection, lambda, -stop*15, maxiter, trace);%use the equivalent representation of eq.(19-20) so that it is not necessary to rewrite the key parts of the codes)
        sl                         = fliplr(sl);                      % for the stability and obtaining the sparse projection of SLE
        eigvector                  = [sl ones(size(sl,1),1)];         % just a trick in dimensionality reduction to achieve the expected number of projections (for compensation)
        for dim = 1:max_dimension
            projection             = eigvector(:,1:dim);              % select ptojection vectore
            YYtrain                = projection'*Ytrain;              % fainal traint sequnce
            YYtest                 = projection'*Ytest;               % fainal test sequnce
            re_rat_SLE_KNN(Kneighbor,stop,dim)= KNN_Classfier(YYtrain', lable_train, YYtest',lable_test, 1) %Kneighbor,stop,
        end
        clear YYtrain YYtest
        k=k+1
    end
end
end

