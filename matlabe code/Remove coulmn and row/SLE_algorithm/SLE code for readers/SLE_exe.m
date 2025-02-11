%Sparse linear embedding framework (similar to SPCA)
%note : preprocessing norm(x)=1 for any x may be very important to increase the performance!!
%this programme is used in the paper:
%Zhihui Lai, Waikeung Wong, Yong Xu, Jian Yang, Jinhui Tang and David
%Zhang, Approximate orthogonal sparse embedding for dimensionality reduction, IEEE Transactions on Neural Networks and Learning Systems, 2016, 27(4): 723-735.
clear;clc; close all;
%  A=imreadYale10080(15,11);
%  B=A(1:4:100,1:4:80,:);
load('F:\thesis\database\yalefaces\yale\Yeal_database')  % load data 
B=Yeal;                                                  % input data to algorithm 
[m,n,p]=size(B);                                         % size data
X=reshape(B,[],p);                                       % convert image to vectore
classnum=15;                                             % number classe database
totlenuminclass=11;                                      % number observation in each class
trainingnuminclass=6;                                    % number traing sample
Kneighbor=6;                                             % number niberhod in knn classification 
form=0;
delta=1;                                                 % constrant for norm 2 in SPCA
energyordim=0.95;
load randvector50by11;                                   % load rand vector for selec random data for train and test sequence
% X=reshapeimageCOLtovector(B);             %X:each column is a pattern vector
k=0;
PCAdim=[0 30 40 40 40 40];
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
                       Xrand(:,ind)=X(:,count);%get the random set
                  end
            end
     
        count=0;ind=0;
        clear gnd Xtrain_label Xtrain
        %% random select train sequence
        for i=1:classnum
          for j=1:trainingnuminclass
               count=(i-1)*totlenuminclass+randvector(j);
               ind=ind+1;
               Xtrain(:,ind)=Xrand(:,count);%get the training samples
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
                   %Xforconstructgraph(:,ind2)=X1DPCA(:,count2);%get the training sample for constructiong the graph
%                    XXrandL(((ind2-1)*col+1):(ind2*col),1:row)=squeeze(X2D(:,:,count2))';%construct the big training matrix for computing the scatter matrix
%                    XXrandR(((ind2-1)*row+1):(ind2*row),1:col)=squeeze(X2D(:,:,count2));%construct the big training matrix for computing the scatter matrix
%                    X2Dtraining(:,:,ind2)=squeeze(X2D(:,:,count2));
                  Xtest(:,ind2)=Xrand(:,count2);
             end
        end
    %%      
%          [XPCA eigenvaluel totleenergy flag]=vectorpcamatrixKL(Xrand,classnum,totlenuminclass,trainingnuminclass,150);   
%         %[eigvector,XPCA]=PCA_KL(Xrand,classnum,totlenuminclass,trainingnuminclass,40);
%         [Ydistancematrix]=vdistancematrix(XPCA,classnum,totlenuminclass,trainingnuminclass);
%         [XPCA eigenvaluel totleenergy flag]=vectorpcamatrixKL(Xrand,classnum,totlenuminclass,trainingnuminclass,PCAdim(trainingnuminclass)); %%%PCA dimention reduction for all image vector
%         ratePIEP29_PCA=NNclassifier(Ydistancematrix,classnum,totlenuminclass,trainingnuminclass);
        %-----------------------
%% pre-dimensionality reduction whit SPCA
        [eigvector, eigvalue, elapse] = PCA_dencai(Xtrain', 40);%input row vector is a sample image, for pre-dimensionality reduction to accelerate the speed in SPCA 
         clear Ytrain Ytest
%                 for dim=10:10:size(eigvector,2)
%                 %projection=zeros(ilineeigenvector,dim);    
                projection=eigvector;%(:,1:dim);               % eigen vector PCA
                Ytrain=projection'*Xtrain;                     % new matrix train
                Ytest=projection'*Xtest;%                      % new matrix test
%                 ratePIEP29_PCA6d10(dim/10)= KNN_Classfier(Ytrain, Xtrain_label, Ytest,Xtest_label, 1);
%                 k=k+1
%            end
        %-------------------------------------
        
        %norm(x)=1 for any x is very important to increase the performance!!90.83
%         for i=1:size(XPCA,2)
%             XPCA(:,i)=XPCA(:,i)/norm(XPCA(:,i));
%         end
%         
%             count=0;ind=0;clear Xforconstructgraph;
%             for i=1:classnum
%                  for j=1:trainingnuminclass
%                        count=(i-1)*totlenuminclass+j;
%                        ind=ind+1;gnd(ind)=i;
%                        Xforconstructgraph(:,ind)=XPCA(:,count);%get the training sample for constructiong the graph
%                  end
%             end
            Xforconstructgraph=Ytrain;
            KK=[3 9 27 trainingnuminclass*classnum-2]
            
        for Kneighbor=1:4%:5%:100%2:2:20
            sup=0;%0:unsupervised, 1:supervised
            r=0.01;
            W= Cons_W_lle(Xforconstructgraph,KK(Kneighbor));%construct the LLE matrix as the same one of the Science paper
            
            %------------constructing within-class reconstruction matrix__________
%             Wwin=zeros(classnum*trainingnuminclass,classnum*trainingnuminclass);
%                 count=0;ind=0;clear Xclass
%                 for i=1:classnum
%                      Xclass=Xforconstructgraph(:,(i-1)*trainingnuminclass+1:(i-1)*trainingnuminclass+trainingnuminclass);
%                      %WW=full(W_lle(Xclass,trainingnuminclass-1,0.01));
%                      Wwin(((i-1)*trainingnuminclass+1):i*trainingnuminclass,((i-1)*trainingnuminclass+1):i*trainingnuminclass)= full(W_lle(Xclass,Kneighbor,0.01));%my OK;using this SNPE is best
%                 end 
%                  W=Wwin-diag(diag(Wwin));
          
            ImW=eye(size(W))-W;%construct the matrix as I-W
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
           
            % ---------------------------------perform SPCA----------------------------
            for stop=1:3  %size(XX,2)-1%stop=36:92.5%
              %[sl sv pcal pcav plots] = spca(XX, [], K, lambda, -stop, maxiter, trace);
              
              Nprojection=150;
              [sl sv pcal pcav plots] = spca([], XX, Nprojection, lambda, -stop*15, maxiter, trace);%use the equivalent representation of eq.(19-20) so that it is not necessary to rewrite the key parts of the codes)
            %[sl theta rss] = slda(Xtr, Ytr, lambda, stop, maxiter, 1);%sl is the sparse projection (matrix) in each colomn
%            s=size(sl,2);
%            for i=1:s
%                eigvector(:,i)=sl(:,s-i);
%            end

            sl=fliplr(sl);  % for the stability and obtaining the sparse projection of SLE
            eigvector=[sl ones(size(sl,1),1)];%just a trick in dimensionality reduction to achieve the expected number of projections (for compensation)
            
            
            % Project data onto the sparse directions (dim=2)
            %DC = X*sl;

                for dim=1:35
%                             %projectmat=zeros(classnum*trainingnuminclass,dl);%dimension_pca * d_graph_select
%                             projectmat=zeros(size(eigvector,1),Genergyordim*5);
%                             projectmat(:,:)=eigvector(:,1:Genergyordim*5);
%                             Ygraphembed=projectmat'*XPCA;
%                             %Ygraphembed=projectmat'*Xrand;%(PCA_dimension*dl)'*(PCA_dimension* classnum*totlenuminclass)
%                              %(48*18) * (48*165)=18*165
%                             [Ydistancematrix]=vdistancematrix(Ygraphembed,classnum,totlenuminclass,trainingnuminclass);
%                             %[Yvpcaclassifierrate(floor(alfa/10)+1),mod(Kneighbor,5)]=NNclassifier(Ydistancematrix,classnum,totlenuminclass,trainingnuminclass);
%                             %rateYaleCMVM3to6rand((Kneighbor)/5,trainingnuminclass,traintime,(Genergyordim))=NNclassifier(Ydistancematrix,classnum,totlenuminclass,trainingnuminclass);% highest rate(3,2)=93.3(yale)
%                             ratePIEP29_PCAaSLE2to6d5(Kneighbor,stop,trainingnuminclass,traintime,Genergyordim)=NNclassifier(Ydistancematrix,classnum,totlenuminclass,trainingnuminclass);% highest rate(3,2)=93.3(yale)
%                            
%                            %trainingnuminclass,traintime,i,Kneighbor,trainingnuminclass,stop,trainingnuminclass,traintime,trainingnuminclass,traintime,
                    projection=eigvector(:,1:dim);
                    YYtrain=projection'*Ytrain;
                    YYtest=projection'*Ytest;%
                    rateYale_PCAaSLE3to6d1(Kneighbor,stop,trainingnuminclass,traintime,dim)= KNN_Classfier(YYtrain', Xtrain_label', YYtest',Xtest_label', 1);%Kneighbor,stop,
                end  % for itreitve dimention
                clear YYtrain YYtest
                k=k+1
            end      % for whit stop criterion
%                 [Ydistancematrix2]=vdistancematrix(XPCA,classnum,totlenuminclass,trainingnuminclass);
%                 rateYale_PCA2to6d1rand10(trainingnuminclass,traintime)=NNclassifier(Ydistancematrix2,classnum,totlenuminclass,trainingnuminclass);% highest rate(3,2)=93.3(yale)
        end          % for itreitive k nearest neighborhood
     end             % for number iterative in time
end                  % for number trainin sample each classe

%%
% Yplot=eigvector(:,1:2)'*Xrand;
% plot(Yplot(1,1:48),Yplot(2,1:48),'.',Yplot(1,49:96),Yplot(2,49:96),'+',Yplot(1,96:end),Yplot(2,96:end),'o')
%% rand vector
% % [~,index]=sort(rand(11,50));
% % randvector50by11=index';
% % save randvector50by11 randvector50by11
%[a,b,c,d,e]=size(rateYale_PCAaSLE3to6d1);
