%%                     IN NAME OF ALLAH
function[y,zc,zs,rmse_ckr,rmse_skr]=rand_recon(image, missing_pixel)
img=double(image);
% create a random sampling map
y=img;
rand('state', 0);
I = double(rand(size(y)) > missing_pixel);
y =I.*img;
% pilot estimation by second order classic kernel regression
h = 2.3;    % the global smoothing parameter
ksize = 15; % the kernel size
[zc, zx1c, zx2c] = ckr2_irregular(y, I, h, ksize);
% root mean square error
error = img - zc;
rmse_ckr = sqrt(mymse(error(:)));

% obtain orientation information
wsize = 9;   % the size of local analysis window
lambda = 1;  % the regularization for the elongation parameter
alpha = 0.1; % the structure sensitive parameter
C = steering(zx1c, zx2c, I, wsize, lambda, alpha);

% applying steering kernel regression for the irregularly sampled image
h = 2.3;
[zs, zx1s, zx2s] = skr2_irregular(y, I, h, C, ksize);
% root mean square error
error = img - zs;
rmse_skr = sqrt(mymse(error(:)));

% display irregularly downsampled image
% figure; imagesc(y.*I); colormap(gray); axis image;
% title('The irregularly downsampled image (7.7% of the original pixels are missing)');
% figure; imagesc(zc); colormap(gray); axis image;
% title(['The reconstructed image by second order classic kernel regression, RMSE=', num2str(rmse_ckr)]);
% figure; imagesc(zs); colormap(gray); axis image;
% title(['The reconstructed image by second order steering kernel regression, RMSE=', num2str(rmse_skr)]);
end