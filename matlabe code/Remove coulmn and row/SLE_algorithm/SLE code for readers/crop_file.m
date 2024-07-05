%%           IN THE NAIME AF ALLAH
clear; clc; close all;
% load('F:\thesis\orl_faces\orl_database1.mat')
% data= orl_database1;
load('F:\thesis\database\yalefaces\yale\Yeal_database.mat')
data= Yeal;
% load('F:\thesis\database\FEI\frontal\FACENEW.mat')
% data=FACEA;
%%
x=data;
[n,m,N] = size(x);
%% parameter for crop image
row     = n/2;         % row between 0 and n
column  = m/2;         % column between 0 and m 
%% parameter for crop-SPCA function
delta_r=inf;           % delta_r = inf ; 
stop_r =-312;          % stop_r  = m-10 ; m  number of row
delta_l=inf;           % delta_r = inf;   
stop_l=-233;           % stop__l = n-10;  n number of column
%% crop-spca image function 
[index_l,index_r,B_left,A_left,B_right,A_right]=crop_spca(x,delta_r,delta_l,stop_r,stop_l);

%% crop-image
crop_img=x(sort(index_l(1,1:row)),sort(index_r(1,1:column)),:);  
s_l=sort(index_l(1,1:row));
s_r=sort(index_r(1,1:column));
%%
figure('NAME','crop image')
for i=1:N
subplot(1,2,1);imshow(x(:,:,i),[0 255]);
title(['original image ' 'dimention =' num2str(n),'*', num2str(m)])
subplot(1,2,2);imshow(crop_img(:,:,i),[0 255]);
title(['crop image ' 'dimention =' num2str(floor(row)),'*', num2str(floor(column) )])
pause(.4)
end