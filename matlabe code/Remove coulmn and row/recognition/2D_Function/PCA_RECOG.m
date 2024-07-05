%% IN ThE NAME OF ALLAH
function [re_rat_pca_KNN] = PCA_RECOG (x_train,lable_train,lable_test,x_test,max_dimension,k,Distance_mark)
%%
re_rat_pca_KNN = zeros(max_dimension,1);  % recognition rate matrix use K-NN classification
[~,~,n_tr]     = size(x_train);
[~,~,n_te]     = size(x_test);
x_tr           = reshape(x_train,[],n_tr)';  % convert matrix train to vector
x_te           = reshape(x_test,[],n_te)';   % convert mtrix  test to vector
% centering train data
mu             = mean(x_tr);                 % calculate mean of train data
x_tr           = x_tr - ones(n_tr,1)*mu;     % centering train data
% centering test data
x_te           = x_te - ones(n_te,1)*mu;     % centering test data
%% CALL PCA FUNCTION
[coeff,score,~]= pca(x_tr,'Centered',false,'Economy',false,'NumComponents',max_dimension);
x_tr_dim       = score;            % reduce dimention for train data
x_te_dim       = x_te*coeff;       % reduce dimenson  for test data

%% K Nearest neighborhood classification
for dimension = 1:max_dimension
    re_rat_pca_KNN(dimension,1) = KNN_Classfier(x_tr_dim(:,1:dimension),lable_train,x_te_dim(:,1:dimension),lable_test,k,Distance_mark);
end
end


