%% IN THE NAME OF AALH
function[B_left,A_left,B_right,A_right]=BSPCA(x,delta_r,delta_l,stop_r,stop_l )
[m,n,N]                    = size(x);               % size of observation
d_row                      = n-2;
d_col                      = m-2;
x                          = double(x);
mu                         = mean(x,3);             % calculate mean af image
x                          = x-repmat(mu,[1 1 N]);  % centering each  observation
for ii=1:N
    d                      = sqrt(sum(x(:,:,ii).^2));
    d(d == 0)              = 1;
    x(:,:,ii)              = x(:,:,ii)./(ones(m,1)*d);
end
clear ii;
s_r1                       = zeros([n n N]);
s_c1                       = zeros([m m N]);
for i=1:N
    s_r1(:,:,i)            = x(:,:,i)'*x(:,:,i);        % alculate  row covariance for each observation
    s_c1(:,:,i)            = x(:,:,i)*x(:,:,i)';        % alculate  column covariance for each observation
end
s_row                      = mean(s_r1,3);              % sum row covariance matrix
s_col                      = mean(s_c1,3);              % sum column covariance matrix
[B_left,A_left,~,~,~,~]    = spca([],s_col,d_col,delta_l,stop_l); %use spca algorithm
[B_right,A_right,~,~,~,~]  = spca([],s_row,d_row,delta_r,stop_r);
end

