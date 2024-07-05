%% IN THE NAME OF ALLAH
function [re_rat] = towd_KN_classificer(x_tr,lable_train,x_test,lable_test,Number_row_daimention,Number_coulmn_daimention)
re_rat            = zeros([Number_row_daimention,Number_coulmn_daimention]);
[~,~,M]           =  size(x_test);
for row_dimention = 1:Number_row_daimention
    for  coulmn_dimention =  1: Number_coulmn_daimention
        p                                     = NN2_matrix(x_tr(1:row_dimention,1:coulmn_dimention,:),lable_train,x_test(1:row_dimention,1:coulmn_dimention,:));
        nu_recogniz                           = numel(find(lable_test-p == 0));
        re_rat(row_dimention,coulmn_dimention)= nu_recogniz/(M);
    end
end
end