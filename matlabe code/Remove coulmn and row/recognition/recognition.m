%%
clc; clear; close all;
%%
d = 30;
numberclass = 40;
numbertrainingsample = 9;
numbereachclass = 10;
numertestsample = numbereachclass-numbertrainingsample;
%%
load('orl_database1.mat')
data_orgin = double(orl_database1);
data_orgin = reshape(data_orgin,[],400)';
load('data_crop.mat')
data_crop = double(data_crop);
data_crop = reshape(data_crop,[],400)';
%imshow(crop_img(:,:,1))
%% train and test sequnce
con=1;con1=1;
for i=1:numberclass
    for j=1:numbertrainingsample
        x_tr_or(con,:)  = data_orgin((j+(i-1)*numbereachclass),:);      % creat train sequence
        x_tr_cr(con,:)  = data_crop((j+(i-1)*numbereachclass),:);
        lable_tr(con,1) = i;                                             % creat lable for train sequence
        con=con+1;                                                       % counter
    end
    for jj=1:numertestsample
        x_te_or(con1,:) = data_orgin((numbertrainingsample+jj)+(i-1)*numbereachclass,:); % creat test sequence
        x_te_cr(con1,:) = data_crop((numbertrainingsample+jj)+(i-1)*numbereachclass,:);
        lable_te(con1,1)= i;                                           % creat lable for test sequence
        con1=con1+1;                                                   % counter
    end
end
clear i j jj con con1;
%%
%% PCA algorithm
%training
[p_or,dor]                      = size(x_tr_or);             % dimenstion of matric
[p_cr,dcr]                      = size(x_tr_cr);             % convert matrix image to vector
mu_or                           = mean(x_tr_or);
mu_cr                           = mean(x_tr_cr);             % calculate mean aof matrix
x_tr_or                         = x_tr_or-ones(p_or,1)*mu_or;% centring data matrix
x_tr_cr                         = x_tr_cr-ones(p_cr,1)*mu_cr;
%Rows of X correspond to observations and columns correspond to variables
[coeff_or,score_or,latent_or]   = pca(x_tr_or,'Centered',false,'Algorithm','svd','NumComponents',d);
[coeff_cr,score_cr,latent_cr]   = pca(x_tr_cr,'Centered',false,'Algorithm','svd','NumComponents',d);
x_tr_or                         = score_or;
x_tr_cr                         = score_cr;
% % %% test pca
% %  x_=x_trp*coeff';
% %  x_hat=x_+ones(p_trp,1)*mu;
% % imshow(reshape(x_hat(1,:),[112 92]),[0 256]);
%% test sequnce
[pp_or,~]     = size(x_te_or);
[pp_cr,~]     = size(x_te_cr);
x_te_or       = x_te_or-ones(pp_or,1)*mu_or;
x_te_or       = x_te_or*coeff_or;
x_te_cr       = x_te_cr-ones(pp_cr,1)*mu_cr;
x_te_cr       = x_te_cr*coeff_cr;
%%%% NN nearest neighborhood classification
for dd=1:d
    p_or            = NN(x_tr_or(:,1:dd),lable_tr,x_te_or(:,1:dd));
    nu_recogniz_or  = numel(find(lable_te-p_or == 0));
    re_rat_or(1,dd) = nu_recogniz_or/(numberclass*numertestsample);
    
    p_cr            = NN(x_tr_cr(:,1:dd),lable_tr,x_te_cr(:,1:dd));
    nu_recogniz_cr  = numel(find(lable_te-p_cr == 0));
    re_rat_cr(1,dd) = nu_recogniz_cr/(numberclass*numertestsample);
    
end
%% plot
figure (1)
hold on
plot(re_rat_or,'-*')
plot(re_rat_cr,'-*')
legend('or','cr')