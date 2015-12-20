function BER=computeCost(y,tX,beta)

    predictLog     = sigmoid(tX * beta);
    pred           = round(predictLog);
    matrix         = zeros(2,2);
     for j = 1:length(y)
        
        switch y(j)
            
            case 1
                if pred(j) == 1
                    matrix(1,1) = matrix(1,1) + 1;
                else
                    matrix(1,2) = matrix(1,2) + 1;
                end
            case 0
                if pred(j) == 0
                     matrix(2,2) =  matrix(2,2) + 1;
                else
                    matrix(2,1) =  matrix(2,1) + 1;
                end
        end 
     end
     
      BER = (matrix(1,1)/(matrix(1,1) + matrix(1,2)) + ...
             matrix(2,2)/(matrix(2,2) + matrix(2,1)))/2;
      BER = 1-BER;
% 	cost=-y'*tX*beta+sum(log(ones(instanceNum,1)+exp(tX*beta)));
% 	cost=cost/instanceNum;
end