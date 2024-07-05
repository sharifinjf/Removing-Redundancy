%%
close all
clc
clear all
N=10;  %number of img
dm=112; %number of max reduce dimention 
delta=100; %coefficient norm 2
stop=100000; %coefficient norm 1
maxSteps=3000;
convergenceCriterion=10^-9;
verbose=0;
for i=1:N
img1=imread(['C:\Users\a\Desktop\icc\databeas\s11\' num2str(i) '.pgm']);
% img=imresize(img,[92 90]);
img1=double(img1);
img(i,:)=img1(:);
end
%%


n=size(img,1);
[img,mu,d]=normalize(img);
% img =(img-ones(n,1)*mu)./sqrt(ones(n,1)*d);
Gram=img'*img;
%%
 [B,AA,SD,L,D,paths] = spca(img, Gram, dm, delta, stop, maxSteps, convergenceCriterion, verbose);


for dd=2:2:dm
y=img*B(:,1:dd);
img_=y*AA(:,1:dd)';
erore(1,dd/2)=norm(img_-img);
recognitionrate(1,dd/2)=1-(norm(erore(1,dd/2))/norm(img));
end

plot(2:2:dm,recognitionrate,'-*')