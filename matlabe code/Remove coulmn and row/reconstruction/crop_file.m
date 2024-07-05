%%           IN THE NAIME AF ALLAH
clear; clc; close all;
%%
%1.orginal Yeal 2.manually crop Yeal 3.ORL 4.coil_20 5.FEI 6.digits data
 n      = 4;
[x]     = load_data(n);       %call function load data
xx1     = load_data(4);
%%
[n,m,N] = size(x);
%% parameter for crop image
row         = 80;            % row between 0 and n
column      = 80;            % column between 0 and m
%% parameter for crop-SPCA function
delta_r     = inf;            % delta_r = inf ;
stop_r      = -(m-15);        % stop_r  = m-10 ; m  number of row
delta_l     = inf;            % delta_r = inf;
stop_l      = -(n-15);        % stop__l = n-10;  n number of column
%% crop-spca image function
[index_l,index_r,B_left,A_left,B_right,A_right]=crop_spca(x,delta_r,delta_l,stop_r,stop_l);
%% crop-image
crop_img     = x(sort(index_l(1,1:row)),sort(index_r(1,1:column)),:);             % crop image
r_crop_imag  = x(sort(index_l(1,(row+1):end)),sort(index_r(1,(column+1):end)),:); % removed row and coulmn in imag
rr_crop_imag = x;
rr_crop_imag(sort(index_l(1,(row+1):end)),sort(index_r(1,(column+1):end)),:) = 0;  % equle zero row and colmn removing
% % s_l = sort(index_l(1,1:row));
% % s_r = sort(index_r(1,1:column));
%%

figure('NAME','orginal image and zero piexel')
[ha, pos] = tight_subplot(2,5,[.01 .01],[.3 .1],[0.1 0.1]); % function for adjust beetwen subplot
for i = 1:5
    axes(ha(i)); imshow(x(:,:,i),[0 255]);
    axes(ha(i+5));imshow(rr_crop_imag(:,:,i),[0 255]);
end
%%
figure('NAME','crop image')
[ha, pos] = tight_subplot(2,10,[.01 .01],[.35 .35],[0.001 0.01]);  % function for adjust beetwen subplot
for i=1:10
    axes(ha(i));imshow(crop_img(:,:,i),[0 255]);
    axes(ha(i+10));imshow(xx1(:,:,i),[0 255]);
end
%%
% % figure('NAME','remove image')
% % [ha, pos] = tight_subplot(1,5,[.01 .01],[.2 .2],[0.001 0.01]); % function for adjust beetwen subplot
% % for i=1:4
% %     axes(ha(i));imshow(r_crop_imag(:,:,i),[0 255]);
% %     suptitle('remove image');
% % end
%%