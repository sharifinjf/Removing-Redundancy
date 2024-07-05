function [B_right,A_right,m]=towDSPCAright(database,d,lambda2,lambda1,convergedCriterion)
img=database;
K=d;
%% Mean calculation 
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
[~,~,V]= svd(R_cov_mat_img);
A=V(:,1:K);
B_right=zeros(c,K);
%% calculation loop of A and B_right 
STEP=zeros(1,K);
for  k=1:K
    step=0;
    criterion=10;
    while (criterion>convergedCriterion)
    step=step+1;
    Bk_right_old=B_right(:,k);
    f=@(B) (B'*R_cov_mat_img*B)-(2*(A(:,k))'*R_cov_mat_img*B)+(lambda2*B'*B)+(lambda1*sum(abs(B)));
    fun=@(B) f(B);
    B0=zeros(c,1);
    option=optimoptions('fminunc','Algorithm','quasi-newton');
    [B,fval]=fminunc(fun,B0,option);
    B_norm=sqrt(sum(B.^2));
    if B_norm==0
        B_norm=1;
    end
    normal_B=B./B_norm;
    B_right(:,k)=normal_B;
    %update A
    t=R_cov_mat_img*B_right(:,k);
    S=t-(A(:,1:k-1)*(A(:,1:k-1)'*t));
    A_norm=sqrt(S'*S);
     if A_norm==0
        A_norm=1;
    end
    A(:,k)=S/A_norm;
    %converged?
    criterion=sum((Bk_right_old-B_right(:,k)).^2);
    end
    STEP(1,k)=step;
end
A_right=A;
end