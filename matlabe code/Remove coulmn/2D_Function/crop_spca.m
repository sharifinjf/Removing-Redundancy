%%                  IN TEH NAME OF ALLAH
function[index_l,index_r,B_left,A_left,B_right,A_right]=crop_spca(x,delta_r,delta_l,stop_r,stop_l )
[m,n,N]                   = size(x);               % size of observation
d_row                     = n;
d_col                     = m;
x                         = double(x);
mu                        = mean(x,3);             % calculate mean af image
 for kk = 1:N
     x(:,:,kk)            = x(:,:,kk)-mu;
 end
% % x                         = x-repmat(mu,[1 1 N]);  % centering each  observation
for ii=1:N
    d                     = sqrt(sum(x(:,:,ii).^2));
    d(d == 0)             = 1;
    x(:,:,ii)             = x(:,:,ii)./(ones(m,1)*d);
end
clear ii;
s_r1                       = zeros([n n]);
s_c1                       = zeros([m m]);
for i=1:N
    s_r1                   = s_r1+(x(:,:,i)'*x(:,:,i));     % alculate  row covariance for each observation
    s_c1                   = s_c1+(x(:,:,i)*x(:,:,i)');     % alculate  column covariance for each observation
end
s_row                      = s_r1/N;                % sum row covariance matrix
s_col                      = s_c1/N;                % sum column covariance matrix
[B_left,A_left,~,~,~,~]    = spca([],s_col,d_col,delta_l,stop_l); %use spca algorithm
[B_right,A_right,~,~,~,~]  = spca([],s_row,d_row,delta_r,stop_r);
z_bl                       = sum(B_left'==0);        % number zerose in each row B_left
[~,index_l]                = sort(z_bl);             % sort number zerose B_left matrix in each row
z_br                       = sum(B_right'==0);       % number zerose in each row B_right
[~,index_r]                = sort(z_br);             % sort number zerose B_right matrix in each row
end
