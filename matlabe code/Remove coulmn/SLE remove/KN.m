%% IN THE NAME OF ALLAH
function INDEX =KN(x,k)
[~,m] = size(x);
INDEX = zeros(k,m);
for NumberObesr =1:m
    for  Number = 1:m
         dif(1,Number)     = sum((x(:,NumberObesr)-x(:,Number)).^2);
    end
     [value,index]             = sort(dif);
     INDEX(:,NumberObesr)  = index(1,2:k+1);
end

end