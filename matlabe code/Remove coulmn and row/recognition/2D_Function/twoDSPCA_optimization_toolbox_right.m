 clc;clear;close all;
%%     
 K=30;                           %number dimention
 convergedCriterion=1e-3;    
 lambda2=.00001;                 %constant for norm 2
 lambda1=5000;                   %constant for norm 1
%% Mean calculation 
 load('Yale_database1.mat')          %load database
 img=double(Yale_database1);
 [r,c,n]=size(img);
 m=1/n*sum(img,3);                   %mean database
 M=ones(r,c,n).*m;                   %produce matrix M whit size database 
 img=img-M;                          %minus total mean af each image 
 R_cov=zeros(c,c);       
 L_cov=zeros(r,r);
for i=1:n
 R_cov=R_cov+img(:,:,i)'*img(:,:,i);  %calculation  right covariance matrix
 L_cov= L_cov+img(:,:,i)*img(:,:,i)'; %calculation  right covariance matrix
end

R_cov_mat_img=R_cov./r;
L_cov_mat_img= L_cov./c;
[U,D,V]= svd(L_cov_mat_img);
A_Left=V(:,1:K);
B_Left=zeros(r,K);
%% calculation loop of A_Left and B_Left 
STEP=zeros(1,K);
for  k=1:K
    step=0;
    criterion=10;
    while (criterion>convergedCriterion)
        step=step+1;
    Bk_Left_old=B_Left(:,k);
    f=@(B) (B'*L_cov_mat_img*B)-(2*(A_Left(:,k))'*L_cov_mat_img*B)+(lambda2*B'*B)+(lambda1*sum(abs(B)));
    fun=@(B) f(B);
    B0=zeros(r,1);
    option=optimoptions('fminunc','Algorithm','quasi-newton');
    [B,fval]=fminunc(fun,B0,option);
    B_norm=sqrt(sum(B.^2));
    if B_norm==0
        B_norm=1;
    end
    normal_B=B./B_norm;
    B_Left(:,k)=normal_B;
    %update A_Left
    t=L_cov_mat_img*B_Left(:,k);
    S=t-(A_Left(:,1:k-1)*(A_Left(:,1:k-1)'*t));
    A_norm=sqrt(S'*S);
     if A_norm==0
        A_norm=1;
    end
    A_Left(:,k)=S/A_norm;
    %converged?
    criterion=sum((Bk_Left_old-B_Left(:,k)).^2);
    end
    STEP(1,k)=step;
end


%% compression rate calculation
d_img=B_Left'*img(:,:,1);
r_img=A_Left*d_img;
r_img=r_img+m;
imshow(r_img,[0 255])
