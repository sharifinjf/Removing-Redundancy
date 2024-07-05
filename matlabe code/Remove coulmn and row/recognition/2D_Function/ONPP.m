%%        IN THE NAME OF ALLAH
function [re_rat_oNPP_KNN] = ONPP(x_train,lable_train,lable_test,x_test,MaxDim,k_ONPP,k,Distance_mark)
[~,~,M]        = size(x_test);
[~,~,N]        = size(x_train);
img_test       = reshape(x_test,[],M)';
img            = reshape(x_train,[],N)';
% % n              = size(img,1);
mu             = mean(img);
img            = img-ones(N,1).*mu;
img_test       = img_test-ones(M,1).*mu;
img_test       = img_test';
img            = img';                               % compute k nearest neighbor
[~,Nn]         = size(img);
X2             = sum(img.^2,1);
distancee      = repmat(X2,Nn,1)+repmat(X2',1,Nn)-2*(img)'*img;
tol            = 1e-3;
[~,index] = sort(distancee);
neighborhood   = index(2:(1+k_ONPP),:);
W = zeros(k_ONPP,Nn);
              %compute the weight of k nearest
for ii=1:Nn
   z           = img(:,neighborhood(:,ii))-repmat(img(:,ii),1,k_ONPP); % shift ith pt to origin
   C           = z'*z;                                                 % local covariance
   C           = C + eye(k_ONPP,k_ONPP)*tol*trace(C);                  % regularlization (K>D)
   W(:,ii)     = C\ones(k_ONPP,1);                                     % solve Cw=1
   W(:,ii)     = W(:,ii)/sum(W(:,ii));                                 % enforce sum(w)=1
end
 M = sparse(1:Nn,1:Nn,ones(1,Nn),Nn,Nn,4*k_ONPP*Nn); 
 for ii=1:N
   w = W(:,ii);
   jj = neighborhood(:,ii);
   M(ii,jj) = M(ii,jj) - w';
 end
              %compute the ONNP
mm   =  img*M*(M)'*(img)';
 
options.disp  = 0; options.isreal  = 1; options.issym = 1;
[eigenvectorOnpp,eigenvalueonpp]   = eigs(mm,MaxDim+1,'lm');
[aaa,~]                            = sort(eigenvalueonpp);
[~,bbbb]                           = sort(aaa(MaxDim+1,:));
eigenvectorOnpp                    = eigenvectorOnpp(:,bbbb(1,2:MaxDim+1));
                 %compute the LLE
% m                                  = M*M';
% [eigenvectorlle,eigenvaluelle]     = eigs(m,d+1,0,options);
% [aaa,bbb]                          = sort(eigenvaluelle);
% [aaaa,bbbb]                        = sort(aaa(d+1,:));
% eigenvectorlle                     = eigenvectorlle(:,2:d+1)'*sqrt(N);
% % % ylle                           = img*eigenvectorlle;
 Y_train_onpp                        = eigenvectorOnpp'*img;
 Y_test_Onpp                         = eigenvectorOnpp'*img_test ; 
 %% classification
for dim = 1:MaxDim
 re_rat_oNPP_KNN(dim,1)= KNN_Classfier(Y_train_onpp(1:dim,:)', lable_train, Y_test_Onpp(1:dim,:)',lable_test,k,Distance_mark);

end
end