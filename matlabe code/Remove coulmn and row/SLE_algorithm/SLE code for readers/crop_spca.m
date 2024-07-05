%%                  IN TEH NAME OF ALLAH
% clc; clear; close all;

% delta=2.5;
% stop=30;

% load('orl_database1.mat')
% x=orl_database1;
% data=orl_database1;
function[index_l,index_r,B_left,A_left,B_right,A_right]=crop_spca(x,delta_r,delta_l,stop_r,stop_l )
[m,n,N]         = size(x);               % size of observation 
d_row           = n-2;
d_col           = m-2;
x               = double(x);                
mu              = mean(x,3);             % calculate mean af image 
x               = x-repmat(mu,[1 1 N]);  % centering each  observation
for ii=1:N
d = sqrt(sum(x(:,:,ii).^2));
d(d == 0) = 1;
x(:,:,ii) = x(:,:,ii)./(ones(m,1)*d);
end
clear ii;
s_r1            = zeros([n n N]);
s_c1            = zeros([m m N]);
for i=1:N
    s_r1(:,:,i) = x(:,:,i)'*x(:,:,i);         % alculate  row covariance for each observation
    s_c1(:,:,i) = x(:,:,i)*x(:,:,i)';         % alculate  column covariance for each observation
end
s_row           = mean(s_r1,3);          % sum row covariance matrix 
s_col           = mean(s_c1,3);          % sum column covariance matrix
[B_left,A_left,~,~,~,~]      = spca([],s_col,d_row,delta_l,stop_l); %use spca algorithm
[B_right,A_right,~,~,~,~]    = spca([],s_row,d_col,delta_r,stop_r);
z_bl                         = sum(B_left'==0);   %number zerose in each row B_left
[~,index_l]                  = sort(z_bl);        %sort number zerose B_left matrix in each row
z_br                         = sum(B_right'==0);  %number zerose in each row B_right
[~,index_r]                  = sort(z_br);        %sort number zerose B_right matrix in each row
end
% % crop_img=data(sort(index_l(1,1:end/2)),sort(index_r(1,1:end/2)),:);    % crop image
% % %% plot
% % figure('NAME','test')
% % y = B_left'*x(:,:,400)*B_right;
% % x_hat = B_left*y*B_right';
% % x_hat=x_hat.*(ones(m,1)*d);
% % x_hat = x_hat+mu;
% % imshow(x_hat,[0 255])
% % 
% % %%
% % figure('NAME','crop image')
% % for i=1:100
% %     imshow(crop_img(:,:,i),[0 255])
% % end
