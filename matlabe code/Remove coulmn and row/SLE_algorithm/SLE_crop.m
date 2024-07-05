%%                              IN THE NAIME OF ALLAH
%Sparse linear embedding framework (similar to SPCA)
%note : preprocessing norm(x)=1 for any x may be very important to increase the performance!!
%this programme is used in the paper:
%Zhihui Lai, Waikeung Wong, Yong Xu, Jian Yang, Jinhui Tang and David
%Zhang, Approximate orthogonal sparse embedding for dimensionality reduction, IEEE Transactions on Neural Networks and Learning Systems, 2016, 27(4): 723-735.
clear;clc; close all;

load('F:\thesis\database\yalefaces\yale\Yeal_database')  % load data 
X=Yeal;                                                  % input data to algorithm                           
classnum=15;                                             % number classe database
totlenuminclass=11;                                      % number observation in each class
trainingnuminclass=6;                                    % number traing sample
delta=1;                                                 % constrant for norm 2 in SPCA
load randvector50by11;                                   % load rand vector for selec random data for train and test sequence
k=0;
rateYale_SPCA4to6d1rand10=[];                            
for trainingnuminclass=3:6
     for traintime=1:10
            randvector=zeros(1,totlenuminclass);
            randvector=randvector50by11(traintime,:);    %randvector_UCIwine10by48(traintime,:);
             count=0;ind=0;
            clear Xrand gnd i j;
            %% produce rand matrix x fro each class
            for i=1:classnum
                  for j=1:totlenuminclass
                       count=(i-1)*totlenuminclass+randvector(j);
                       ind=ind+1;
                       Xrand(:,:,ind)=X(:,:,count);%get the random set
                  end
            end
        count=0;ind=0;
        clear gnd Xtrain_label Xtrain
        %% random select train sequence
        for i=1:classnum
          for j=1:trainingnuminclass
               count=(i-1)*totlenuminclass+randvector(j);
               ind=ind+1;
               Xtrain(:,:,ind)=Xrand(:,:,count);       % get the training samples
               gnd(ind)=i;
               Xtrain_label(ind)=i;
          end
        end   
         count2=0;
         ind2=0;
         clear Xforconstructgraph XXrand Xtest_label Xtest
         %% random select test sequence
        for i=1:classnum
             for j=1+trainingnuminclass:totlenuminclass
                   count2=(i-1)*totlenuminclass+j;
                   ind2=ind2+1;
                   Xtest_label(ind2)=i;
                   Xtest(:,:,ind2)=Xrand(:,:,count2);
             end
        end
%%------------------------------- crop image --------------------------%%
 %% parameter for crop-SPCA function
[n,m,p]=size(Xtrain);
[~,~,p1]=size(Xtest);
delta_r = inf;           % delta_r = inf ; 
stop_r  = -(m-12);       % stop_r  = m-10 ; m  number of row
delta_l = inf;           % delta_r = inf;   
stop_l  = -(n-12);       % stop__l = n-10;  n number of column
%% crop-spca image function 
[index_l,index_r,B_left,A_left,B_right,A_right]=crop_spca(Xtrain,delta_r,delta_l,stop_r,stop_l); 
%% parameter for calculate recognition rate
d_r =115;                % new dimension for row crop image
d_c =200;                % new dimension for column crop image
crop_img = Xtrain(sort(index_l(1,1:d_r)),sort(index_r(1,1:d_c)),:);     % select best row and column
%% train and test sequnce                                               % convert image to vectore
Xtrain=reshape(crop_img,[],p);
Xtest =Xtest(sort(index_l(1,1:d_r)),sort(index_r(1,1:d_c)),:);
Xtest=reshape(Xtest,[],p1);                                             % convert image to vectore
%% pre-dimensionality reduction whit SPCA
        [eigvector, eigvalue, elapse] = PCA_dencai(Xtrain', 40);%input row vector is a sample image, for pre-dimensionality reduction to accelerate the speed in SPCA 
         clear Ytrain Ytest
  
                projection=eigvector;%(:,1:dim);               % eigen vector PCA
                Ytrain=projection'*Xtrain;                     % new matrix train
                Ytest=projection'*Xtest;%                      % new matrix test
%% LLE and calculate Graph matrix 
            Xforconstructgraph=Ytrain;
            KK=[3 9 27 trainingnuminclass*classnum-2]
            
        for Kneighbor=1:4%:5%:100%2:2:20
            sup=0;                                             %0:unsupervised, 1:supervised
            r=0.01;
            W= Cons_W_lle(Xforconstructgraph,KK(Kneighbor));   %construct the LLE matrix as the same one of the Science paper
            
            %------------constructing within-class reconstruction matrix__________
            ImW=eye(size(W))-W;                                %construct the matrix as I-W
            XX=Xforconstructgraph*ImW'*ImW*Xforconstructgraph';%construct the special scatter matric for SLE used for SPCA regression (it is the equivalent representation of the formula in the paper)
            %--------------------------------SPCA preparition------------------------
%             XX=(Xforconstructgraph*ImW')';clear projectmat
%             XX = normalize(XX);%X is :each row is a normalized vector.
%             [n p] = size(XX);
%             K = size(XX,2);%6
              %you can set the following parameters 
            lambda = 1000;
            maxiter = 50;
            trace = 1;
%% SLE algorithm           
            % ---------------------------------perform SPCA----------------------------
            for stop=1:2  %size(XX,2)-1%stop=36:92.5%
                Nprojection=150;
                [sl,AA,sv,pcal,pcav,plots] = spca([], XX, Nprojection, lambda, -stop*15, maxiter, trace);%use the equivalent representation of eq.(19-20) so that it is not necessary to rewrite the key parts of the codes)
                sl=fliplr(sl);                             % for the stability and obtaining the sparse projection of SLE
                eigvector=[sl ones(size(sl,1),1)];         % just a trick in dimensionality reduction to achieve the expected number of projections (for compensation)
%            parpool(3)
                for dim=1:36
                    projection=eigvector(:,1:dim);         % select ptojection vectore 
                    YYtrain=projection'*Ytrain;            % fainal traint sequnce
                    YYtest=projection'*Ytest;              % fainal test sequnce
                    rateYale_PCAaSLE3to6d1(Kneighbor,stop,trainingnuminclass,traintime,dim)= KNN_Classfier(YYtrain', Xtrain_label', YYtest',Xtest_label', 1);%Kneighbor,stop,
                end  % for itreitve dimention
                clear YYtrain YYtest
                k=k+1 
            end      % for whit stop criterion
        end          % for itreitive k nearest neighborhood
     end             % for number iterative in time
end                  % for number trainin sample each classe

%% rand vector
% % [~,index]=sort(rand(11,50));
% % randvector50by11=index';
% % save randvector50by11 randvector50by11
%  [a,b,c,d,e]=size(rateYale_PCAaSLE3to6d1);
%% plot
save resultcrop1 rateYale_PCAaSLE3to6d1
