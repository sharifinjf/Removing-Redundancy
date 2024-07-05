%%                 IN the NAME OF ALLAH
clc; clear ; close all;
%% loud data
% 1.orginal Yeal 2.manually crop Yeal 3.ORL 4.coil_20 5.FEI 6.digits data 7.AR
 n   = 1;
[x]  = load_data(n);   %call function load data
 m   = 2;
[x1] = load_data(m);   %call function load data

%----------------------------------orl-------------------------------------
% % % class number = 40
% % % observition each class = 10
%--------------------------------- FEI-------------------------------------
% % % class number = 200
% % % observition each class = 14
% --------------------------------Yeal-------------------------------------
% % % class number = 15
% % % observition each class = 11
%--------------------------------coil_20-----------------------------------
% % % class number = 20
% % % observition each class = 72
%-------------------------------- digits--------------------------------------
% % % x_train = x{1,1} label_train = {1,2}
% % % x_test  = x{2,1] label_test  = {2,2}
%--------------------------------AR-------------------------------------------
% % % class number = 126
% % % observition each class = 11
% ----------------------------------------------------------------------------
%% produce train and test sequnce
numberclass          = 15;
numbertrainingsample = 5;
numbereachclass      = 11;
numertestsample      = numbereachclass-numbertrainingsample;
% call function train and test
if n~=6  % n=6 is digits database not need train_set function
    [x_train,lable_train,lable_test,x_test]=train_test(x,numberclass,numbertrainingsample,numbereachclass);
    [x_train_m_c,lable_train_m_c,lable_test_m_c,x_test_m_c]=train_test(x1,numberclass,numbertrainingsample,numbereachclass);
end
% digits test and train
if n == 6
    x_train     = x{1,1};
    lable_train = x{1,2};
    x_test      = x{2,1};
    lable_test  = x{2,2};
end
%% parameter for crop-SPCA function
[n,m,~]               = size(x_train);
delta_r               = inf;              % delta_r = inf
stop_r                = (-173);           % stop_r  = m-10 ; -m<stop_r<-1
delta_l               = inf;              % delta_r = inf
stop_l                = (-163);            % stop_l = n-10  -n<stop_l<-1
%% parameter for S-B2DPCA
delta_r_sb2dpca       = inf;
delta_l_sb2dpca       = inf;
stop_r_sb2dpca        = -(m-2);           % stop_r_sb2dpca  = m-10 ; -m<stop_r<-1
stop_l_sb2dpca        = -(n-3);           % stop_l_sb2dpca  = n-10  -n<stop_l<-1
%% crop-spca image function
[index_l,index_r,~,~,~,~] = crop_spca(x_train,delta_r,delta_l,stop_r,stop_l);
%% parameter for calculate recognition rate
d_r      = 100;          % new dimension for row crop image          % 1 < d_r < n
d_c      = 100;           % new dimension for column crop image       % 1 < d_c < m
d2dpca   = 40;           % max dimention for 2DPCA                   % 1 < d2dpca < m;
d2dpca_c = 40;           % max dimention for 2DPCA aftar crop image  % 1 < d2dpca_c < d_c
d_co_B   = 40;           % max dimention coulmn for B-2DPCA          % 1 < d_co_B < n
d_ro_B   = 40;           % max dimention row for B-2DPCA             % 1 < d_ro_B < m
d_c_co_B = 40;           % max dimention coulmn for B-2DPCA after crop image  % 1 < d_c_co_B < d_r
d_c_ro_B = 40;           % max dimention row for B-2DPCA after crop image     % 1 < d_c_co_B < d_c
%% crop image
crop_img = x_train(sort(index_l(1,1:d_r)),sort(index_r(1,1:d_c)),:);     % select best row and column for train data
%% train and test sequnce
x_train_c  = crop_img;
x_test_c   = x_test(sort(index_l(1,1:d_r)),sort(index_r(1,1:d_c)),:);     %  select best row and column for test data
%% 2DPCA  for manualy crop data
 [~,~,N]    = size(x_train_m_c);                 % size    manually  crop train data
 [n,~,M]    = size(x_test_m_c);                  % size manually crop  test data
% % x_tr_2dpca_m_c = zeros([n d2dpca N]);        % train data that reduced dimention whit 2dpca
% % x_te_2dpca_m_c = zeros([n d2dpca M]);        % test  data that reduced dimention whit 2dpca
% % [w] = D2PCA(x_train_m_c);                    % call  function 2DPCA
% % % calculate new train data
% % for jj=1:N
% %     x_tr_2dpca_m_c(:,:,jj) = x_train_m_c(:,:,jj)*w(:,1:d2dpca);
% % end
% % % calculate new test data
% % for ii=1:M
% %     x_te_2dpca_m_c(:,:,ii) = x_test_m_c(:,:,ii)*w(:,1:d2dpca);
% % end
% % clear jj ii ;
%% 2DPCA for crop data
% % [m1,~,N]     = size(x_train_c);            % size  crop train data
% % [~,~,M]      = size(x_test_c);             % size  crop test data
% % x_tr_c_2dpca = zeros([m1 d2dpca_c N]);  % crop train data that reduced dimention whit 2dpca
% % x_te_c_2dpca = zeros([m1 d2dpca_c M]);  % crop test data that reduced dimention  whit 2dpca
% % [w1]= D2PCA(x_train_c);                 % call function 2DPCA
% % % calculate new crop train data
% % for jj=1:N
% %     x_tr_c_2dpca(:,:,jj) = x_train_c(:,:,jj)*w1(:,1:d2dpca_c);
% % end
% % % calculate new crop test data
% % for ii=1:M
% %     x_te_c_2dpca(:,:,ii) = x_test_c(:,:,ii)*w1(:,1:d2dpca_c);
% % end
% % clear jj ii;
%% B2DPCA for original image
x_tr_b2pca    = zeros([d_co_B d_ro_B N]);  % train data that reduced dimention whit B2dpca
x_te_b2pca    = zeros([d_co_B d_ro_B M]);  % test  data that reduced dimention whit B2dpca
[w_col,w_row] = B2DPCA(x_train);           % call function B2DPCA
% calculate new train data  whit B2DPC
for jj=1:N
    x_tr_b2pca(:,:,jj) = w_col(:,1:d_co_B)'*x_train(:,:,jj)*w_row(:,1:d_ro_B);
end
% calculate new  train data whit B2DPC
for ii=1:M
    x_te_b2pca(:,:,ii) = w_col(:,1:d_co_B)'*x_test(:,:,ii)*w_row(:,1:d_ro_B);
end
clear ii jj;
%% B2DPCA for crop image
x_tr_c_b2pca      = zeros([d_c_co_B d_c_ro_B N]);    % crop train data that reduced dimention whit B2dpca
x_te_c_b2pca      = zeros([d_c_co_B d_c_ro_B M]);    % crop test  data that reduced dimention whit B2dpca
[w_col_c,w_row_c] = B2DPCA(x_train_c);        % call function B2DPCA
% calculate new crop train data  whit B2DPC
for jj=1:N
    x_tr_c_b2pca(:,:,jj) = w_col_c(:,1:d_c_co_B)'*x_train_c(:,:,jj)*w_row_c(:,1:d_c_ro_B);
end
% calculate new crop test data whit B2DPC
for ii=1:M
    x_te_c_b2pca(:,:,ii) = w_col_c(:,1:d_c_co_B)'*x_test_c(:,:,ii)*w_row_c(:,1:d_c_ro_B);
end
clear ii jj;
%% S-B2DPCA for orginal image
% % x_tr_s_b2pca    = zeros([d_co_B d_ro_B N]);  % train data that reduced dimention whit S-B2dpca
% % x_te_s_b2pca    = zeros([d_co_B d_ro_B M]);  % test  data that reduced dimention whit S-B2dpca
% % [~,~,B_left,A_left,B_right,A_right] = crop_spca(x_train,delta_r_sb2dpca,delta_l_sb2dpca,stop_r_sb2dpca,stop_l_sb2dpca);  % call function S-B2DPCA
% % % calculate new train data  whit S-B2DPC
% % for jj=1:N
% %     x_tr_s_b2pca(:,:,jj) = B_left(:,1:d_co_B)'*x_train(:,:,jj)*B_right(:,1:d_ro_B);
% % end
% % % calculate new  train data whit S-B2DPC
% % for ii=1:M
% %     x_te_s_b2pca(:,:,ii) = B_left(:,1:d_co_B)'*x_test(:,:,ii)*B_right(:,1:d_ro_B);
% % end
%% B2DPCA for manuly crop data
x_tr_m_c_b2pca        = zeros([d_c_co_B d_c_ro_B N]);    % crop train data that reduced dimention whit B2dpca
x_te_m_c_b2pca        = zeros([d_c_co_B d_c_ro_B M]);    % crop test  data that reduced dimention whit B2dpca
[w_col_m_c,w_row_m_c] = B2DPCA(x_train_m_c);             % call function B2DPCA
% calculate new crop train data  whit B2DPC
for jj=1:N
    x_tr_m_c_b2pca(:,:,jj) = w_col_m_c(:,1:d_c_co_B)'*x_train_m_c(:,:,jj)*w_row_m_c(:,1:d_c_ro_B);
end
% calculate new crop test data whit B2DPC
for ii=1:M
    x_te_m_c_b2pca(:,:,ii) = w_col_m_c(:,1:d_c_co_B)'*x_test_m_c(:,:,ii)*w_row_m_c(:,1:d_c_ro_B);
end
clear ii jj;

%% classification:nearest neighborhood
%% 2DPCA original image
% % re_rat_2dpca_m_c       = zeros(1,d2dpca);
% % for dd=1:d2dpca
% %     p                  = NN2_matrix(x_tr_2dpca_m_c(:,1:dd,:),lable_train,x_te_2dpca_m_c(:,1:dd,:));
% %     nu_recogniz        = numel(find(lable_test-p == 0));
% %     re_rat_2dpca_m_c(1,dd) = nu_recogniz/(numberclass*numertestsample);
% % end
% % clear dd p nu_recogniz ;
%% 2DPCA crop image
% % re_rat_c_2dpca           = zeros([1 d2dpca_c]);
% % for dd = 1:d2dpca_c
% %     p                    = NN2_matrix(x_tr_c_2dpca(:,1:dd,:),lable_train,x_te_c_2dpca(:,1:dd,:));
% %     nu_recogniz          = numel(find(lable_test-p == 0));
% %     re_rat_c_2dpca(1,dd) = nu_recogniz/(numberclass*numertestsample);
% % end
% % clear dd p nu_recogniz;
%% B2DPCA original image
% % re_rat_b2dpca           = zeros([1,d_co_B]);
% % for dd=1:d_co_B
% %     p                   = NN2_matrix(x_tr_b2pca(1:dd,1:dd,:),lable_train,x_te_b2pca(1:dd,1:dd,:));
% %     nu_recogniz         = numel(find(lable_test-p == 0));
% %     re_rat_b2dpca(1,dd) = nu_recogniz/(numberclass*numertestsample);
% % end
% % clear dd p nu_recogniz;
%% B2DPCA for crop image
re_rat_c_b2dpca           = zeros([1,d_co_B]);
for dd=1:d_co_B
    p                     = NN2_matrix(x_tr_c_b2pca(1:dd,1:dd,:),lable_train,x_te_c_b2pca(1:dd,1:dd,:));
    nu_recogniz           = numel(find(lable_test-p == 0));
    re_rat_c_b2dpca(1,dd) = nu_recogniz/(numberclass*numertestsample);
end
clear dd p nu_recogniz;
%% S-B2DPCA for orginal image
% % re_rat_s_b2dpca           = zeros([1,d_co_B]);
% % for dd=1:d_co_B
% %     p                     = NN2_matrix(x_tr_s_b2pca(1:dd,1:dd,:),lable_train,x_te_s_b2pca(1:dd,1:dd,:));
% %     nu_recogniz           = numel(find(lable_test-p == 0));
% %     re_rat_s_b2dpca(1,dd) = nu_recogniz/(numberclass*numertestsample);
% % end
% % clear dd p nu_recogniz;

%%  B2DPCA for manuly crop data
re_rat_m_c_b2dpca           = zeros([1,d_co_B]);
for dd=1:d_co_B
    p                       = NN2_matrix(x_tr_m_c_b2pca(1:dd,1:dd,:),lable_train_m_c,x_te_m_c_b2pca(1:dd,1:dd,:));
    nu_recogniz             = numel(find(lable_test_m_c-p == 0));
    re_rat_m_c_b2dpca(1,dd) = nu_recogniz/(numberclass*numertestsample);
end
clear dd p nu_recogniz;
%%

%%
%% plot
% % figure ('NAME','2DPCA')
% % hold on
% % plot(re_rat_2dpca_m_c,'-*')
% % plot(re_rat_c_2dpca,'-*')
% % legend('2DPCA manually crop','2DPCA automatic crop image')
% % xlabel('dimension')
% % ylabel('recognition rate')
% % hold off
%%
% % figure ('NAME','B2DPCA')
% % hold on
% % plot(re_rat_b2dpca,'-*')
% % plot(re_rat_c_b2dpca,'-*')
% % plot(re_rat_s_b2dpca,'-*')
% % legend('B2DPCA','B2DPCA after crop image','S-B2DPCA')
% % xlabel('dimension')
% % ylabel('recognition rate')
% % hold off
%%
figure ('NAME','manuly and automatic crop')
hold on 
plot(re_rat_m_c_b2dpca,'-*')
plot(re_rat_c_b2dpca,'-*')
legend('manuly crop','automatic crop')
xlabel('dimension')
ylabel('recognition rate')

%%
%  figure('NAME','crop image')
%  for i=1:N
%  subplot(1,2,1); imshow(x_train(:,:,i),[0 255]);   title('original image')
%  subplot(1,2,2); imshow(x_train_c(:,:,i),[0 255]); title('crop train image')
%  pause(.6)
%  end
%  %%
%  figure('NAME','test crop image')
%   for i=1:M
%   imshow(x_test_c(:,:,i),[0 255]); title('crop test image')
%  pause(.6)
%  end
%%
% svmStructure = fitcsvm(reshape(x_tr_c_2dpca(1:5,1:5,:),[],60)',lable_train');
% grop         = ClassificationSVM(svmStructure,reshape(x_te_c_b2pca(1:5,1:5,:),[],105)');
%aa = multisvm(reshape(x_tr_c_2dpca(1:10,1:10,:),[],60)',lable_train,reshape(x_te_c_b2pca(1:10,1:10,:),[],105)');

