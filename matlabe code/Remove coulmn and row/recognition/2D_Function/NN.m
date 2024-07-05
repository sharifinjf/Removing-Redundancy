%% IN THE NAME OF ALLAH
% use  for 2D matrix
function[predict]     = NN(x_train,lable_train,x_test)
[n_te,~]              = size(x_test);
[n_tr,~]              = size(x_train);
predict               = zeros(1,n_te);
% d                     = zeros(1,n_tr);
for jj = 1:n_te
    d                = sqrt(sum(((repmat(x_test(jj,:),n_tr,1)- x_train).^2)'));   % find minimum distance
    [~,index]        = min(d);                                                 
    predict(jj,1)    = lable_train(index);
end
end
