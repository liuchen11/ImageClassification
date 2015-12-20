% -------------- logisticRegression (GRADIENT DESCENT) ------------
% Function: beta = logisticRegression(y,tX,alpha)
% Purpose : calculate the regression parameter beta using GD method
% Input   : y      --- the target vector of the given data set
%           tX     --- the N-by-(D+1) matrix constructed from data
%           alpha  --- the GD step size
% Output  : beta   --- parameter beta optimizing the cost
% -------------- C. LIU & M. ZHAO ---------------------------------
function beta = logisticRegression(y,tX,alpha)
    
    instanceNum= size(tX,1);
    featureNum = size(tX,2);
    beta       = zeros(featureNum,1);
    
    maxIter = 1000000;
    
    for i = 1:maxIter
        
        gradient = tX' * (sigmoid(tX * beta) - y);
        % disp(gradient'*gradient);
        % disp(beta);
        gradient=gradient/instanceNum;
        %Very small gradient show the converage of the model
%          gradient' * gradient
        
        if gradient' * gradient < 1e-3
        
            meg  = sprintf('Logisitic Regression Converges at Iteration %d\n',i);
        
            disp(meg);
        
            return;
        end
        beta = beta - alpha * gradient;
        % if mod(i,100)==0
        %     disp(gradient'*gradient);
        % end
    end
    
    meg = sprintf('Logistic Regression Ends\n');
    
    disp(meg);
end

