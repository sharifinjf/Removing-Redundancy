%%                 IN the NAME OF ALLAH
% distanc is ferbenuce norm
% use for 3D matrix x(n,m,p)
function[predict]    = NNf_matrix(x_train,lable_train,x_test)
[~,~,p_te]           = size(x_test);            % size data test
[~,~,p_tr]           = size(x_train);           % size data train
predict              = zeros(p_te,1);           % predit marix
x_test               = reshape(x_test,[],p_te); % matrix imag convert to vector; each column one observition
x_train              = reshape(x_train,[],p_tr);
for jj=1:p_te
    d                = sqrt(sum((repmat(x_test(:,jj),1,p_tr)-x_train).^2));   % find minimum distance
    [~,index]        = min(d);
    predict(jj,1)    = lable_train(index);
end
end