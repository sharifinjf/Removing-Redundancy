%% IN ThE NAME OF ALLAH
clc; clear ; close all;
%%
addpath('F:\thesis\matlabe code\remove coulmn and row\recognition\2D_Function')
addpath('F:\thesis\matlabe code\remove coulmn and row\recognition\Function')
addpath('F:\thesis\matlabe code\remove coulmn and row\recognition\data')

%%
MaxDim = 40;     % maximum dimention for recognition
Distance_mark = 'Cos'; % Distance_mark:['Euclidean', 'L2'| 'L1' | 'Cos']
%% loud data
% 1.orginal Yeal 2.manually crop Yeal 3.ORL 4.coil_20 5.FEI 6.digits data
% 7.orginal AR   8.crop manully AR
n      = 1;
[x]    = load_data(n);   %call function load data
select  = 1;              %'select = 1' use crop image function

%% produce train and test sequnce
numberclass          = 15;
numbertrainingsample = 6;
numbereachclass      = 11;
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
    for  d_r  = 100:-1:50       % new dimension for row crop image          % 1 < d_r < n
        for  d_c  = 100:-1:50       % new dimension for column crop image       % 1 < d_c < m
            %% crop image
            crop_img    = x_train(sort(index_l(1,1:d_r)),sort(index_r(1,1:d_c)),:);     % select best row and column for train data
            %% train and test sequnce
            x_train1  = crop_img;
            x_test1   = x_test(sort(index_l(1,1:d_r)),sort(index_r(1,1:d_c)),:);     %  select best row and column for test data
        end
        %% dimensionality reduction
        %% ------------------------------------------ PCA-----------------------------------
        %re_rat_pca_KNN  = PCA_RECOG (x_train,lable_train,lable_test,x_test,MaxDim,k,Distance_mark);
        %% -------------------------------------------SLE-----------------------------------
        re_rat_SLE_KNN  = SLE(x_train1,lable_train,lable_test,x_test1,MaxDim,numbertrainingsample,numberclass);
        %%
        %re_rat_ONPP_KNN = ONPP(x_train,lable_train,lable_test,x_test,MaxDim,k_ONPP,k,Distance_mark);
        total_SLE_Yale_aut{d_r,d_c} = re_rat_SLE_KNN;
        %%
        aa = ['dr = ',num2str(d_r) ,'   ','dc = ',num2str(d_c)];
        disp(aa)
    end
end
%%
save total_SLE_Yale_aut total_SLE_Yale_aut
%%
% % load('aut_re_rat_SLE_KNN.mat')
% % a1(1,:) = re_rat_SLE_KNN(3,3,1:40);
% % load('mn_re_rat_SLE_KNN.mat')
% % a2(1,:) = re_rat_SLE_KNN(3,3,1:40);
% % figure
% % hold on
% % plot(1:40,a1(1,:),'-*')
% % plot(1:40,a2(1,:),'-*')
% % legend('aut','man')