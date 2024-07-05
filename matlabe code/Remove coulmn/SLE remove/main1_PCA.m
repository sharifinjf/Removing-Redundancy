%% IN ThE NAME OF ALLAH
clc; clear ; close all;
%%
%addpath 2D_Function
addpath Function
addpath('D:\university\sbu\thesis\matlabe code\remove coulmn\recognition\data')
addpath imm3897
%%
MaxDim        = 40;     % maximum dimention for recognition
k             = 1;      % k for K Nearest neighborhood classification
Distance_mark = 'L2';   % Distance_mark:['Euclidean', 'L2'| 'L1' | 'Cos'] 
Kneighbor     = 6;      % k for K SLE remove
%% remove SLE parametr 
remove        = 15000;
lambda        = inf;
stop          = -14000;
dm            = 50;   % in SLE remove
%% remove SPCA parametr
remove_SPCA   = remove;
delta_SPCA    = inf;
stop_SPCA     = -15000;
dm_SPCA       = 50; 
%% loud data
% 1.orginal Yeal 2.manually crop Yeal 3.ORL 4.coil_20 5.FEI 6.digits data
% 7.orginal AR   8.crop manully AR
n                    = 4;
[x]                  = load_data(n);   %call function load data
%% produce train and test sequnce
numberclass          = 20;
numbertrainingsample = 40;
numbereachclass      = 72;
numertestsample      = numbereachclass-numbertrainingsample;
% call function train and test
%% SLE REMOVE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[x_train,lable_train,lable_test,x_test] = train_test(x,numberclass,numbertrainingsample,numbereachclass); %  train and test
%% Train
[n,m,N]                = size(x_train);
basic_img_train        = reshape(x_train,[],N);             
img                    = normalize(basic_img_train')';
B                      = SLE_fOR_REMOVE(img,Kneighbor,stop,lambda,dm);
%% remve train
p = n*m;
number_zeros                      = sum(B'==0);   %calculate the number of zerose in each row
[value,index]                     = sort(number_zeros); 
Xtrain1                           = basic_img_train;
Xtrain1(index(p:-1:p-remove+1),:) = [];             % zero select columns 
Xtrain1                           = Xtrain1';
basic_img_train                   = basic_img_train';
%% Test
[~,~,M]                           = size(x_test);
basic_img_test                    = reshape(x_test,[],M);             
Xtest1                            = basic_img_test;
Xtest1(index(p:-1:p-remove+1),:)  = [];
Xtest1                            = Xtest1';
basic_img_test                    = basic_img_test';
% dimensionality reduction
%% PCA %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[x_train2,x_test2]      = RemoveSPCA(x_test,x_train,remove_SPCA,delta_SPCA,stop_SPCA,dm_SPCA);
%% ------------------------------------------ PCA-----------------------------------
re_rat_pca_KNN_out_SPCA = PCA_RECOG (x_train2,lable_train,lable_test,x_test2,MaxDim,k,Distance_mark);
re_rat_pca_KNN_out_SLE  = PCA_RECOG (Xtrain1,lable_train,lable_test,Xtest1,MaxDim,k,Distance_mark);
re_rat_pca_KNN_man      = PCA_RECOG (basic_img_train,lable_train,lable_test,basic_img_test,MaxDim,k,Distance_mark);
%% -------------------------------------------SLE-----------------------------------
re_rat_SLE_KNN_out_SPCA = SLE(x_train2,lable_train,lable_test,x_test2,MaxDim,numbertrainingsample,numberclass);
re_rat_SLE_KNN_out_SLE  = SLE(Xtrain1,lable_train,lable_test,Xtest1,MaxDim,numbertrainingsample,numberclass);
re_rat_SLE_KNN_man      = SLE(basic_img_train,lable_train,lable_test,basic_img_test,MaxDim,numbertrainingsample,numberclass);

%%
 figure('Name','PCA compare')
 hold on
 plot(re_rat_pca_KNN_out_SPCA*100,'-*')
 plot(re_rat_pca_KNN_out_SLE*100,'-*')
 plot(re_rat_pca_KNN_man*100,'-^')
 legend({['after removed ', num2str(remove) ,' pixel with SPCA'],...
 ['after removed ', num2str(remove) ,' pixel with SLE'],'orginal image'}...
 ,'FontSize',12,'Location','southeast')
 ylabel('Recognition rate (%)','FontSize',14)
 xlabel('Dimention','FontSize',14)
 grid on
 box on

%%
figure('Name','SLE compare')
hold on
re_rat_SLE_out_SPCA(1,1:40)= re_rat_SLE_KNN_out_SPCA(4,4,1:40);
re_rat_SLE_out_SLE(1,1:40) = re_rat_SLE_KNN_out_SLE(4,4,1:40);
re_rat_SLE_man(1,1:40)     = re_rat_SLE_KNN_man(4,4,1:40);
plot(re_rat_SLE_out_SPCA*100,'-*')
plot(re_rat_SLE_out_SLE*100,'-*')
plot(re_rat_SLE_man*100,'-^')

legend({['after removed ', num2str(remove) ,' pixel with SPCA'],...
 ['after removed ', num2str(remove) ,' pixel with SLE'],'orginal image'}...
 ,'FontSize',12,'Location','southeast')
ylabel('Recognition rate (%)','FontSize',14)
xlabel('Dimention','FontSize',14)
grid on
box on

