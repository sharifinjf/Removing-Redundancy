clear; clc; close all;
addpath 2D_Function
addpath data
%% Load data
% 1.orginal Yeal 2.manually crop Yeal 3.ORL 4.coil_20 5.FEI 6.digits data 7.AR
% 8.manully crop AR
n    = 8;
[x]  = load_data(n);   %call function load data
%%
AR_crop_manually = x;
con = 1;                            
[n,m,N] = size(x);
for cuntr_class =1:100
 for   cunter_sample =1:10
   %  AR1(:,:,con) = AR(:,:,(20*(cuntr_class-1))+ cunter_sample); 
   %  AR2(:,:,con) = AR(:,:,(20*(cuntr_class-1))+ (10+cunter_sample));
     AR_crop_manually1(:,:,con) = AR_crop_manually(:,:,(20*(cuntr_class-1))+ cunter_sample);
     AR_crop_manually2(:,:,con) = AR_crop_manually(:,:,(20*(cuntr_class-1))+ (10+cunter_sample));
     con = con +1;
end
end
%save AR1 AR1
%save AR2 AR2
 save AR_crop_manually1 AR_crop_manually1
 save AR_crop_manually2 AR_crop_manually2
 %% 
 for i=1:1:10
     subplot(2,5,i);
     imshow(AR_crop_manually1(:,:,(10*8)+i))
 end