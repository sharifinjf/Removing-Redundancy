%%                     IN THE NIME OF ALLAH


function[x_train,lable_tr,lable_te,x_test] = train_test(x,n_class,n_train_sample,n_each_class)
[m,n,~]=size(x);
numertestsample=n_each_class-n_train_sample;
x_train = zeros([m n n_train_sample*n_class]);
x_test  = zeros([m n numertestsample*n_class]);
lable_tr= zeros([n_train_sample*n_class 1]);
lable_te= zeros([numertestsample*n_class 1]);
con=1;con1=1;
for i=1:n_class
    for j=1:n_train_sample
        x_train(:,:,con)=x(:,:,(j+(i-1)*n_each_class));     % creat train sequence
        lable_tr(con,1)=i;                                     % creat lable for train sequence
        con=con+1;                                             % counter
    end
    for jj=1:numertestsample
        x_test(:,:,con1)=x(:,:,(n_train_sample+jj)+(i-1)*n_each_class); % creat test sequence
        lable_te(con1,1)=i;                                % creat lable for test sequence
        con1=con1+1;                                       % counter
    end
end
end