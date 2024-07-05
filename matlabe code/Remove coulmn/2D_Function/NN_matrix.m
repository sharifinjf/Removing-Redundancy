%%                 IN the NAME OF ALLAH
% distace use equation (9) in thtis peaper
% [1]J.Yang,D.Zhang,A.F.Frangi,andJ.Y.Yang,“Two-Dimensional PCA: A New Approach to Appearance-Based Face Representation and Recognition,”...
%IEEE Trans.Pattern Anal.Mach.Intell.,vol.26,no.1,pp.131–137,2004.
function[predict] = NN_matrix(x_train,lable_train,x_test)
[~,~,p_te]      = size(x_test);
[~,~,p_tr]      = size(x_train);
dis=zeros(p_tr,p_te);
for i=1:p_te
    for jj=1:p_tr
        d1           = sqrt(sum(sum(x_train(:,:,jj)-x_test(:,:,i)).^2));
        dis(jj,i)    = d1;
    end
end
[~,index]    = min(dis);
predict     = lable_train(index(1,:)');
end