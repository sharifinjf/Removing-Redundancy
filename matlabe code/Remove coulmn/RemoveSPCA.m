%% IN THE NME OF ALLAH
function [x_train_re,x_test_re]  = RemoveSPCA(x_test,x_train,remove,delta,stop,dm)
maxSteps                = 3000;        % number max iteration in SPCA
convergenceCriterion    = 10^-3;       % convergence trreshold in SPCA
verbose                 = 0;
[n,m,N]                 = size(x_train);
basic_img               = reshape(x_train,[],N)';        % convert image to vector
mu                      = mean(basic_img);              % calculate mean of image
img                     = basic_img - ones(N,1)*mu;     % centring 
d                       = sqrt(sum(img.^2));            % calculate of variance  
d(d == 0)               = 1;
img                     = img./(ones(N,1)*d);           % normalization to 1 each column
%Gram                 = img'*img;                     % calculate covariance matrix
%% use toolbax spcasm and compute spca and pca
[B,~,~,~,~,~] = spca(img, [], dm, delta, stop, maxSteps, convergenceCriterion, verbose);
%% zeros number in each row matrix B
p=n*m;
number_zeros                        =  sum(B'==0);                    %calculate the number of zerose in each row
[~,index]                           =  sort(number_zeros); 
image_                              = basic_img;
image_(:,index(p:-1:p-remove+1))    = [];                             % zero select columns 
x_train_re                          = image_;    
%% test
[~,~,N]                             = size(x_test);
x_test_re                           = reshape(x_test,[],N)';
x_test_re(:,index(p:-1:p-remove+1)) = [];
end