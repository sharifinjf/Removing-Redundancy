%% In The NAME OF ALLH
clc; clear; close all;
Kneighbor = 40;
%%
addpath Function
addpath('D:\university\sbu\thesis\matlabe code\remove coulmn\recognition\data')
addpath imm3897
%%
n                    = 4;
[x]                  = load_data(n);  
% %  load('Case (1).mat')
% % x =d3;
[n,m,N]              = size(x);
x_r                  = reshape(x,[],N);
%%
W                  = Cons_W_lle(x_r,Kneighbor);
ImW                = eye(size(W))-W;
%%
figure
imshow(W,[]);
%%
X_new              = ImW*x_r';
X_new_reshape      = reshape(X_new',[n m N]);
%%
figure
subplot(1,2,1);imshow(x(:,:,2),[]);title('Orginal image')
subplot(1,2,2);imshow(X_new_reshape(:,:,2),[]);title('Image *(I-w)')