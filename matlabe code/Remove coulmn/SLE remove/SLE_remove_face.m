%% IN THE NAME OF ALLAH
%%                   IN THE NAME OF ALLAH
close all; clc; clear
%%
addpath Function
addpath('D:\university\sbu\thesis\matlabe code\remove coulmn\recognition\data')
addpath imm3897
%%
remove               = 9000;
max_dimension        = 150;
Kneighbor            = 6;
lambda               = inf;         % coefficient norm 2 in SPCA 
stop                 = -8000;      % coefficient norm 1 in SPCA
maxSteps             = 3000;        % number max iteration in SPCA
convergenceCriterion = 10^-3;       % convergence trreshold in SPCA
verbose = 0;
%% load database and normalization
%  1.orginal Yeal 2.manually crop Yeal 3.ORL 4.coil_20 5.FEI 6.digits data
n                    = 2;
[x]                  = load_data(n);                 %call function load data

%% centring and normalization for SPCA algorithm
[n,m,N]                = size(x);
basic_img              = reshape(x,[],N);             
img                    = normalize(basic_img')';
%% SLE ALGoRITHME
%     input column vector is a sample image
 B  = SLE_fOR_REMOVE(img,Kneighbor,stop,lambda,max_dimension);
%% zeros number in each row matrix B
p = n*m;
number_zeros                     = sum(B'==0);   %calculate the number of zerose in each row
[value,index]                    = sort(number_zeros); 
image_                           = basic_img;
image_(index(p:-1:p-remove+1),:) = 0;             % zero select columns 
sampled_image                    = reshape(image_,[n m N]); % convert vector to image
% sampled_image                    = permute(sampled_image,[2 3 1]);
% % % %%
 [ha, pos] = tight_subplot(2, 4,[.01 .005],[.5 .20],[.35 .3]);
 for number_image = 1:4
     axes(ha(number_image)); imshow(x(:,:,number_image),[])
     axes(ha(number_image+4)); imshow(sampled_image(:,:,number_image),[])
end
