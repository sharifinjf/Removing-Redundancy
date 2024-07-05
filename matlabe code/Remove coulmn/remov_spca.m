%%                   IN THE NAME OF ALLAH
close all; clc; clear
%%
addpath Function
addpath data
addpath imm3897
%%
remove               = 2500;  % number remove pixel of each image
delta                = inf;         % coefficient norm 2 in SPCA 
stop                 = -2000;      % coefficient norm 1 in SPCA
dm                   = 150;         % dimension of SPCA
maxSteps             = 3000;        % number max iteration in SPCA
convergenceCriterion = 10^-3;       % convergence trreshold in SPCA
verbose = 0;
%% load database and normalization
% 1.orginal Yeal 2.manually crop Yeal 3.ORL 4.coil_20 5.FEI 6.digits data
n                    = 1;
[x]                  = load_data(n);                 %call function load data

%% centring and normalization for SPCA algorithm
%x                    = imresize(double(x),[90 90]);
[n,m,N]              = size(x);
basic_img            = reshape(x,[],N)';             % convert image to vector
mu                   = mean(basic_img);              % calculate mean of image
img                  = basic_img - ones(N,1)*mu;     % centring 
d                    = sqrt(sum(img.^2));            % calculate of variance  
d(d == 0)            = 1;
img                  = img./(ones(N,1)*d);           % normalization to 1 each column
%Gram                 = img'*img;                     % calculate covariance matrix
%% use toolbax spcasm and compute spca and pca
[B,AA,var_spca,eigenvector_pca,var_pca,paths] = spca(img, [], dm, delta, stop, maxSteps, convergenceCriterion, verbose);
%% zeros number in each row matrix B
p=n*m;
number_zeros         =  sum(B'==0);                    %calculate the number of zerose in each row
[value,index]        =  sort(number_zeros); 
image_               = basic_img;
image_(:,index(p:-1:p-remove+1)) = 0;                   % zero select columns 
sampled_image        = reshape(image_,[ N n m]);      % convert vector to image
sampled_image        = permute(sampled_image,[2 3 1]);
%%
[ha, pos] = tight_subplot(2, 4,[.01 .005],[.5 .20],[.35 .3]);
for number_image = 1:4
    axes(ha(number_image)); imshow(x(:,:,number_image),[])
    axes(ha(number_image+4)); imshow(sampled_image(:,:,number_image),[])
end
   