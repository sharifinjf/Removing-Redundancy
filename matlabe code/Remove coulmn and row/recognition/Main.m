%% IN THE NAME OF ALLAH
% 
%  PREFORMATTED
%  TEXT
% 
clc; clear; close all;
%%
addpath 2D_Function
addpath data
addpath imm3897
addpath FOptM-share
%%
% 1.orginal Yeal 2.manually crop Yeal 3.ORL 4.coil_20 5.FEI 6.digits data
% 7.orginal AR   8.crop manully AR
n    = 4;
[x]  = load_data(n);   %call function load data

% 'mm = 2' for manually crop image % 'mm = 1' for  auto crop image
mm  = 1; 
disp('load data');
%%
%% produce train and test sequnce
numberclass          = 20;
numbertrainingsample = 6;
numbereachclass      = 72;
numertestsample      = numbereachclass-numbertrainingsample;
% call function train and test
if n~=6  % n=6 is digits database not need train_set function
    [x_train,lable_train,lable_test,x_test]=train_test(x,numberclass,numbertrainingsample,numbereachclass);
end
% digits test and train
if n == 6
    x_train     = x{1,1};
    lable_train = x{1,2};
    x_test      = x{2,1};
    lable_test  = x{2,2};
end
disp('finish select train and test data');
%% parameter for crop-SPCA function
[n,m,N]               = size(x_train);    % size   train data
[~,~,M]               = size(x_test);     % size   test data
delta_r               = inf;              % delta_r = inf
stop_r                = -(m-7);           % stop_r  = m-10 ; -m<stop_r<-1
delta_l               = inf;              % delta_r = inf
stop_l                = -(n-7);            % stop_l = n-10  -n<stop_l<-1
%%
d_r      = 90 ;         % new dimension for row crop image          % 1 < d_r < n
d_c      = 90 ;         % new dimension for column crop image       % 1 < d_c < m
d2dpca   = 40;           % max dimention for 2DPCA                   % 1 < d2dpca < m;
d_co_B   = 40;           % max dimention coulmn for B-2DPCA          % 1 < d_co_B < n
d_ro_B   = 40;           % max dimention row for B-2DPCA             % 1 < d_ro_B < m
%% crop-spca image function
[index_l,index_r,~,~,~,~] = crop_spca(x_train,delta_r,delta_l,stop_r,stop_l);
%% crop image
crop_img = x_train(sort(index_l(1,1:d_r)),sort(index_r(1,1:d_c)),:);     % select best row and column for train data
%% train and test sequnce
x_train_c  = crop_img;
x_test_c   = x_test(sort(index_l(1,1:d_r)),sort(index_r(1,1:d_c)),:);     %  select best row and column for test data
disp('finish croping image')
%% dimensionality reduction algorithm
 if  mm == 1
    x_train = x_train_c;
    x_test  = x_test_c;
    [n,m,N] = size(x_train);
 end
%% ----------------------------------2DPCA--------------------------------------
x_tr_2dpca = zeros([n d2dpca N]);        % train data that reduced dimention whit 2dpca
x_te_2dpca = zeros([n d2dpca M]);        % test  data that reduced dimention whit 2dpca
[w]        = D2PCA(x_train);                    % call  function 2DPCA
% % % calculate new train data
for jj=1:N
    x_tr_2dpca(:,:,jj) = x_train(:,:,jj)*w(:,1:d2dpca);
end
% % %  calculate new test data
for ii=1:M
    x_te_2dpca(:,:,ii) = x_test(:,:,ii)*w(:,1:d2dpca);
end
clear jj ii W ;
Number_coulmn_daimention = d2dpca;
[re_rat_2dPCA] = towdpca_classificer(x_tr_2dpca,lable_train,x_te_2dpca,lable_test,Number_coulmn_daimention);
disp('Finish 2DPCA Algorithm')
%
%
%
%
%
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
[re_rat_BPCA]            = towd_KN_classificer(x_tr_bpca,lable_train,x_te_bpca,lable_test,Number_row_daimention,Number_coulmn_daimention);
disp('Finish B2DPCA Algorithm')
%
%
%
%
%
% %% -------------------------------------- BSPCA -------------------------------------------------------
delta_r_sb2dpca                 = inf;
delta_l_sb2dpca                 = inf;
stop_r_sb2dpca                  = -(m-20);           % stop_r_sb2dpca  = m-10 ; -m<stop_r<-1
stop_l_sb2dpca                  = -(n-30);           % stop_l_sb2dpca  = n-10  -n<stop_l<-1
x_tr_bspca                      = zeros([d_co_B d_ro_B N]);  % train data that reduced dimention whit S-B2dpca
x_te_bspca                      = zeros([d_co_B d_ro_B M]);   % test  data that reduced dimention whit S-B2dpca
[B_left,A_left,B_right,A_right] = BSPCA(x_train,delta_r_sb2dpca,delta_l_sb2dpca,stop_r_sb2dpca,stop_l_sb2dpca);  % call function S-B2DPCA
% % % % % calculate new train data  whit S-B2DPC
for jj=1:N
    x_tr_bspca(:,:,jj) = B_left(:,1:d_co_B)'*x_train(:,:,jj)*B_right(:,1:d_ro_B);
end
% % calculate new  train data whit S-B2DPC
for ii=1:M
    x_te_bspca(:,:,ii) = B_left(:,1:d_co_B)'*x_test(:,:,ii)*B_right(:,1:d_ro_B);
end
clear ii jj;
% % classificer
Number_row_daimention    = d_ro_B;
Number_coulmn_daimention = d_co_B;
[re_rat_BSPCA] = towd_KN_classificer(x_tr_bspca,lable_train,x_te_bpca,lable_test,Number_row_daimention,Number_coulmn_daimention);
disp('Finish SB2DPCA Algorithm')
% % %
% % %
% % %
% % %
% % %
%% -----------------------------------------N-2DPCA----------------------------
x_tr_N_2dpca            = zeros([n d2dpca N]);     % train data that reduced dimention whit 2dpca
x_te_N_2dpca            = zeros([n d2dpca M]);     % test  data that reduced dimention whit 2dpca
w                       = PCA2_Nuclear(x_train,m); % call  function N_2DPCA
% calculate new train data
for jj=1:N
    x_tr_N_2dpca(:,:,jj) = x_train(:,:,jj)*w(:,1:d2dpca);
end
%  calculate new test data
for ii=1:M
    x_te_N_2dpca(:,:,ii) = x_test(:,:,ii)*w(:,1:d2dpca);
end
clear jj ii w ;
Number_coulmn_daimention = d2dpca;
[re_rat_N_2dPCA] = towdpca_classificer(x_tr_N_2dpca,lable_train,x_te_N_2dpca,lable_test,Number_coulmn_daimention);
disp('Finish N-2DPCA Algorithm')
%% ----------------------------------N-B2DPCA----------------------------------
%
x_tr_N_bpca             = zeros([d_co_B d_ro_B N]);   % train data that reduced dimention whit B2dpca
x_te_N_bpca             = zeros([d_co_B d_ro_B M]);  % test  data that reduced dimention whit B2dpca
[w_col,w_row,iter,obj]  = PCA2_Bilateral_N(x_train,d_co_B,d_ro_B) ; % call function N-BPCA
% calculate new train data  whit N_BDPC
for jj=1:N
    x_tr_N_bpca(:,:,jj) = w_col(:,1:d_co_B)'*x_train(:,:,jj)*w_row(:,1:d_ro_B);
end
% calculate new  train data whit B2DPC
for ii=1:M
    x_te_N_bpca(:,:,ii) = w_col(:,1:d_co_B)'*x_test(:,:,ii)*w_row(:,1:d_ro_B);
end
clear ii jj;
Number_row_daimention    = d_ro_B;
Number_coulmn_daimention = d_co_B;
[re_rat_N_BPCA]          = towd_KN_classificer(x_tr_N_bpca,lable_train,x_te_N_bpca,lable_test,Number_row_daimention,Number_coulmn_daimention);
disp('Finish N-B2DPCA Algorithm')

%
%
%
%
%
%% -------------------------------2DPCAL1--------------------------------------------
x_tr_2dpcal1 = zeros([n d2dpca N]);        % train data that reduced dimention whit 2dpca
x_te_2dpcal1 = zeros([n d2dpca M]);        % test  data that reduced dimention whit 2dpca
[w]        = PCA2DL1(x_train,d2dpca);      % call  function 2DPCAl1
% % % calculate new train data
for jj=1:N
    x_tr_2dpcal1(:,:,jj) = x_train(:,:,jj)*w(:,1:d2dpca);
end
% % %  calculate new test data
for ii=1:M
    x_te_2dpcal1(:,:,ii) = x_test(:,:,ii)*w(:,1:d2dpca);
end
clear jj ii W ;
Number_coulmn_daimention = d2dpca;
[re_rat_2dPCAl1] = towdpca_classificer(x_tr_2dpcal1,lable_train,x_te_2dpcal1,lable_test,Number_coulmn_daimention);
disp('Finish 2DPCAl1 Algorithm')

%
%
%
%

%% ------------------------------- plot ---------------------------------------------
% % figure
% % hold on
% % plot(re_rat_2dPCA,'-*')
% % plot(diag(re_rat_BPCA),'-*')
% % plot(diag(re_rat_BSPCA),'-*')
% % plot(re_rat_N_2dPCA,'-*')
% % plot(diag(re_rat_N_BPCA),'-*')
% % legend('2DPCA','B2DPCA','B2DSPCA','N_2DPCA','N_B2dPCA')
%% ---------------------------------- save-------------------------------------
coil20_aut_crop_result{1,1} = re_rat_2dPCA;
coil20_aut_crop_result{2,1} = re_rat_BPCA;
coil20_aut_crop_result{3,1} = re_rat_BSPCA;
coil20_aut_crop_result{4,1} = re_rat_N_2dPCA;
coil20_aut_crop_result{5,1} = re_rat_N_BPCA;
coil20_aut_crop_result{6,1} = re_rat_2dPCAl1;
save coil20_aut_crop_result coil20_aut_crop_result
%%
% % ORL_mna_crop_result{1,1} = re_rat_2dPCA;
% % ORL_mna_crop_result{2,1} = re_rat_BPCA;
% % ORL_mna_crop_result{3,1} = re_rat_BSPCA;
% % ORL_mna_crop_result{4,1} = re_rat_N_2dPCA;
% % ORL_mna_crop_result{5,1} = re_rat_N_BPCA;
% % ORL_mna_crop_result{6,1} = re_rat_2dPCAl1;
% % save ORL_mna_crop_result ORL_mna_crop_result
% % 
% % 


