%%                 IN the NAME OF ALLAH
% use for 3D matrix x(n,m,p)
function[predict]= NN2_matrix(x_train,lable_train,x_test)
 [~,~,p_te]      = size(x_test);
 [~,~,p_tr]      = size(x_train);
  dis = zeros(p_tr,p_te);
    for i=1:p_te
        for jj=1:p_tr
            d1           = sqrt(sum(sum(x_train(:,:,jj)-x_test(:,:,i)).^2));
            
            d2           = sqrt(sum(sum(x_train(:,:,jj)'-x_test(:,:,i)').^2));
            dis(jj,i)    = min(d1,d2);
        end
    end
[~,index]    = min(dis);
 predict     = lable_train(index(1,:)');
 end