%% IN ThE NAME OF ALLAH
clc; clear ; close all;
%%
addpath 2D_Function
addpath Function
%%
MaxDim = 100;     % maximum dimention for recognition
k = 1;           % k fo K Nearest neighborhood classification
k_ONPP = 5;
Distance_mark = 'Cos'; % Distance_mark:['Euclidean', 'L2'| 'L1' | 'Cos'] 
%% loud data
% 1.orginal Yeal 2.manually crop Yeal 3.ORL 4.coil_20 5.FEI 6.digits data
% 7.orginal AR   8.crop manully AR
% %  n      = 2;
% %  [x]    = load_data(n);   %call function load data
select = 2;              %'select = 1' use crop image function
load('x.mat')
x = imresize(x,[100 80]);
%% produce train and test sequnce
numberclass          = 18;
numbertrainingsample = 10;
numbereachclass      = 20;
numertestsample      = numbereachclass-numbertrainingsample;
% call function train and test
[x_train,lable_train,lable_test,x_test] = train_test(x,numberclass,numbertrainingsample,numbereachclass); %  train and test
%% parameter for crop-SPCA function
[n,m,~]               = size(x_train);
delta_r               = inf;            % delta_r = inf
stop_r                = -(173);         % stop_r  = m-10 ; -m<stop_r<-1
delta_l               = inf;            % delta_r = inf
stop_l                = -(163);         % stop_l = n-10  -n<stop_l<-1
if select ==1
    %% crop-spca image function
    [index_l,index_r,~,~,~,~] = crop_spca(x_train,delta_r,delta_l,stop_r,stop_l);
    %% parameter for calculate recognition rate
    d_r  = 100;           % new dimension for row crop image          % 1 < d_r < n
    d_c  = 100;           % new dimension for column crop image       % 1 < d_c < m
    %% crop image
    crop_img    = x_train(sort(index_l(1,1:d_r)),sort(index_r(1,1:d_c)),:);     % select best row and column for train data
    %% train and test sequnce
    x_train  = crop_img;
    x_test   = x_test(sort(index_l(1,1:d_r)),sort(index_r(1,1:d_c)),:);     %  select best row and column for test data
end
%% dimensionality reduction
%% ------------------------------------------ PCA-----------------------------------
re_rat_pca_KNN  = PCA_RECOG (x_train,lable_train,lable_test,x_test,MaxDim,k,Distance_mark);
%% -------------------------------------------SLE-----------------------------------
% re_rat_SLE_KNN  = SLE(x_train,lable_train,lable_test,x_test,MaxDim,numbertrainingsample,numberclass);
%%
re_rat_ONPP_KNN = ONPP(x_train,lable_train,lable_test,x_test,MaxDim,k_ONPP,k,Distance_mark);


%%
figure
plot(re_rat_pca_KNN)
%%
figure
for i=1:20
   subplot(2,10,i); imshow(x(:,:,(20*17)+i)); 
end
