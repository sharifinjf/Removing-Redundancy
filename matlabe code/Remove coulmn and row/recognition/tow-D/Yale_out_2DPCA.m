%% IN THE NAME OF ALLAH
%  PREFORMATTED
%  TEXT
% 
clc; clear; close all;
%%
addpath('D:\university\sbu\thesis\matlabe code\remove coulmn and row\recognition\2D_Function')
addpath('D:\university\sbu\thesis\matlabe code\remove coulmn and row\recognition\data')
addpath('D:\university\sbu\thesis\matlabe code\remove coulmn and row\recognition\imm3897')
addpath('D:\university\sbu\thesis\matlabe code\remove coulmn and row\recognition\FOptM-share')
%%
% 1.orginal Yeal 2.manually crop Yeal 3.ORL 4.coil_20 5.FEI 6.digits data
% 7.orginal AR   8.crop manully AR
n    = 1;
[x]  = load_data(3);   %call function load data

% 'mm = 2' for manually crop image % 'mm = 1' for  auto crop image
mm  = 1; 
disp('load data');
%%
%% produce train and test sequnce
numberclass          = 40;
numbertrainingsample = 6;
numbereachclass      = 10;
numertestsample      = numbereachclass-numbertrainingsample;
% call function train and test
if n~=6  % n=6 is digits database not need train_set function
    [x_train1,lable_train,lable_test,x_test1]=train_test(x,numberclass,numbertrainingsample,numbereachclass);
end
% digits test and train
if n == 6
    x_train1     = x{1,1};
    lable_train = x{1,2};
    x_test1      = x{2,1};
    lable_test  = x{2,2};
end
% disp('finish select train and test data');
%% parameter for crop-SPCA function
[n,m,N]               = size(x_train1);    % size   train data
[~,~,M]               = size(x_test1);     % size   test data
delta_r               = inf;              % delta_r = inf
stop_r                = -(n-10);           % stop_r  = m-10 ; -m<stop_r<-1
delta_l               = inf;              % delta_r = inf
stop_l                = -(m-10);            % stop_l = n-10  -n<stop_l<-1
%%
%d_r      = n:1:100 ;     % new dimension for row crop image          % 1 < d_r < n
%d_c      = m:1:150 ;     % new dimension for column crop image       % 1 < d_c < m
d2dpca   = 40;           % max dimention for 2DPCA                   % 1 < d2dpca < m;
d_co_B   = 40;           % max dimention coulmn for B-2DPCA          % 1 < d_co_B < n
d_ro_B   = 40;           % max dimention row for B-2DPCA             % 1 < d_ro_B < m
%% crop-spca image function
[index_l,index_r,~,~,~,~] = crop_spca(x_train1,delta_r,delta_l,stop_r,stop_l);
%% crop image
for Number_row =    70:70
for Number_column = 80:1:80
crop_img = x_train1(sort(index_l(1,1:Number_row)),sort(index_r(1,1:Number_column)),:);     % select best row and column for train data
%% train and test sequnce
x_train_c  = crop_img;
x_test_c   = x_test1(sort(index_l(1,1:Number_row)),sort(index_r(1,1:Number_column)),:);     %  select best row and column for test data
disp('finish croping image')
%% dimensionality reduction algorithm
 if  mm == 1
    x_train = x_train_c;
    x_test  = x_test_c;
 end
% % %% -------------------------------B2DPCA-----------------------------------------
x_tr_bpca                = zeros([d_co_B d_ro_B N]);  % train data that reduced dimention whit B2dpca
x_te_bpca                = zeros([d_co_B d_ro_B M]);  % test  data that reduced dimention whit B2dpca
[w_col,w_row]            = B2DPCA(x_train);           % call function B2DPCA
% %  calculate new train data  whit B2DPC
for jj=1:N
    x_tr_bpca(:,:,jj)    = w_col(:,1:d_co_B)'*x_train(:,:,jj)*w_row(:,1:d_ro_B);
end
% % calculate new  train data whit B2DPC
for ii=1:M
    x_te_bpca(:,:,ii)    = w_col(:,1:d_co_B)'*x_test(:,:,ii)*w_row(:,1:d_ro_B);
end
clear ii jj;
Number_row_daimention    = d_ro_B;
Number_coulmn_daimention = d_co_B;
re_rat_BPCA            = towd_KN_classificer(x_tr_bpca,lable_train,x_te_bpca,lable_test,Number_row_daimention,Number_coulmn_daimention);
total_recognition_aut{Number_row,Number_column} = re_rat_BPCA ;
% % disp('Finish B2DPCA Algorithm')
% % disp(['dr = ' ,num2str(Number_row),'  ', 'dc = ' , num2str(Number_column)])
end
end
%%
% save total_recognition_aut total_recognition_aut