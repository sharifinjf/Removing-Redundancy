%% IN THE NAME OF ALLAH
function [re_rat] = towdpca_classificer(x_tr,lable_train,x_test,lable_test,Number_coulmn_daimention)
re_rat            = zeros([1,Number_coulmn_daimention]);
[~,~,M]           =  size(x_test);
    for  coulmn_dimention =  1: Number_coulmn_daimention
        p                          = NN_matrix(x_tr(:,1:coulmn_dimention,:),lable_train,x_test(:,1:coulmn_dimention,:));
        nu_recogniz                = numel(find(lable_test-p == 0));
        re_rat(1,coulmn_dimention) = nu_recogniz/(M);
    end
end