%% IN ThE NAME OF ALLAH
clc; clear ; close all;
%%
addpath 2D_Function
addpath Function
addpath data
addpath imm3897
%%
MaxDim        = 40;     % maximum dimention for recognition
k             = 1;      % k fo K Nearest neighborhood classification
k_ONPP        = 5;
Distance_mark = 'L2'; % Distance_mark:['Euclidean', 'L2'| 'L1' | 'Cos'] 
%% remove spsca parametr
remove        = 9000;
delta         = inf;
stop          = -9000;
dm            = 30;   % in spaca
%% loud data
% 1.orginal Yeal 2.manually crop Yeal 3.ORL 4.coil_20 5.FEI 6.digits data
% 7.orginal AR   8.crop manully AR
n                    = 3;
[x]                  = load_data(n);   %call function load data

%% produce train and test sequnce
numberclass          = 40;
numbertrainingsample = 6;
numbereachclass      = 10;
numertestsample      = numbereachclass-numbertrainingsample;
% call function train and test
[x_train,lable_train,lable_test,x_test] = train_test(x,numberclass,numbertrainingsample,numbereachclass); %  train and test
[n,m,N]             = size(x_train);
[~,~,M]             = size(x_test);
%%
x_tr                = imresize(x_train,[n m]);
x_te                = imresize(x_test,[n m]);
%%
x_tr                = reshape(x_tr,[],N)';
x_te                = reshape(x_te,[],M)';
%%
[x_train1,x_test1]  = RemoveSPCA(x_test,x_train,remove,delta,stop,dm);
%% dimensionality reduction
%% ------------------------------------------ PCA-----------------------------------
 re_rat_pca_KNN_out  = PCA_RECOG (x_train1,lable_train,lable_test,x_test1,MaxDim,k,Distance_mark);
 re_rat_pca_KNN_man  = PCA_RECOG (x_tr,lable_train,lable_test,x_te,MaxDim,k,Distance_mark);
%% -------------------------------------------SLE-----------------------------------
 re_rat_SLE_KNN_out  = SLE(x_train1,lable_train,lable_test,x_test1,MaxDim,numbertrainingsample,numberclass);
 re_rat_SLE_KNN_man  = SLE(x_tr,lable_train,lable_test,x_te,MaxDim,numbertrainingsample,numberclass);
%%

%%
 figure('Name','PCA compare')
 hold on
 plot(re_rat_pca_KNN_out*100,'-*')
 plot(re_rat_pca_KNN_man*100,'-^')
 legend({['after removed ', num2str(remove) ,' pixel'],'orginal image'}...
  ,'FontName','TimeNew Roman','FontSize',12,'Location','southeast')
 ylabel('Recognition rate (%)','FontName','TimeNew Roman','FontSize',14)
 xlabel('Dimention','FontName','TimeNew Roman','FontSize',14)
 grid on
 box on

%%
figure('Name','SLE compare')
hold on
re_rat_SLE_out(1,1:40) = re_rat_SLE_KNN_out(4,1,1:40);
re_rat_SLE_man(1,1:40) = re_rat_SLE_KNN_man(4,1,1:40);
plot(re_rat_SLE_out*100,'-*')
plot(re_rat_SLE_man*100,'-^')
legend({['after removed ', num2str(remove) ,' pixel'],'orginal image'}...
 ,'FontName','TimeNew Roman','FontSize',12,'Location','southeast')
ylabel('Recognition rate (%)','FontName','TimeNew Roman','FontSize',14)
xlabel('Dimention','FontName','TimeNew Roman','FontSize',14)
grid on
box on

