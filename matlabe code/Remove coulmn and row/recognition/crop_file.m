%%           IN THE NAIME AF ALLAH
clear; clc; close all;
addpath 2D_Function
addpath data
addpath imm3897
%% Load data
% 1.orginal Yeal 2.manually crop Yeal 3.ORL 4.coil_20 5.FEI 6.digits data 7.AR
% 8.manully crop AR
n    = 1;
[x]  = load_data(n);   %call function load data
%load('image.mat')
%x    = double(image);
%%
[n,m,N] = size(x);
%% parameter for crop image
row     = 100;           % row between 0 and n
column  = 100;           % column between 0 and m
%% parameter for crop-SPCA function
delta_r = inf;           % delta_r = inf ;
stop_r  = -(150);        % stop_r  = m-10 ; m  number of row
delta_l = inf;           % delta_r = inf;
stop_l  = -(180);        % stop__l = n-10;  n number of column
%% crop-spca image function
[index_l,index_r,B_left,A_left,B_right,A_right]=crop_spca(x,delta_r,delta_l,stop_r,stop_l);
%% crop-image
crop_img     = x(sort(index_l(1,1:row)),sort(index_r(1,1:column)),:);             % crop image
r_crop_imag  = x(sort(index_l(1,(row+1):end)),sort(index_r(1,(column+1):end)),:); % remove row and coulmn in imag
rr_crop_imag = x;
rr_crop_imag(sort(index_l(1,(row+1):end)),:,:) = 0; 
rr_crop_imag(:,sort(index_r(1,(column+1):end)),:) = 0;
% equle zero row and colmn removing
% % s_l = sort(index_l(1,1:row));
% % s_r = sort(index_r(1,1:column));
%%
[aaa,bbb]=imhist(crop_img,256);
%% plot 
a9 = x(:,:,9);
a10= x(:,:,10);
x(:,:,10) =a9;
x(:,:,9)  = a10;

% %  figure('NAME','orginal image')
% % [ha, pos] = tight_subplot(1,5,[.01 .01],[.2 .25],[0.001 0.01]); % function for adjust beetwen subplot
% % for i=1:5
% %      axes(ha(i));imshow(x(:,:,i),[0 255]);
% % % %     suptitle('original image');
% %  end
 %%
 figure('NAME','crop image')
 [ha, pos] = tight_subplot(7,7,[0.01 0],[.001 .001],[0.001 0.001]);  % function for adjust beetwen subplot
 for i=1:49
     axes(ha(i));imshow(crop_img(:,:,i),[0 255]);
   % xle('crop image');
 end
 %%
% %  figure('NAME','remove image')
% %  [ha, pos] = tight_subplot(1,5,[.01 .01],[.2 .2],[0.001 0.01]); % function for adjust beetwen subplot
% %   for i=1:5
% %    axes(ha(i));imshow(r_crop_imag(:,:,i),[0 255]);
% %      suptitle('remove image');
% %  end
% % %%
 figure('NAME','image with zero pixel')
 [ha, pos] = tight_subplot(6,5,[.01 .01],[.2 .2],[0.001 0.01]); % function for adjust beetwen subplot
for i=1:30
    axes(ha(i));imshow(rr_crop_imag(:,:,i),[0 255]);
     suptitle('image with zero pixel');
 end

%%
%