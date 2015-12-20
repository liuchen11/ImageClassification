% -------------- sigmoid function -----------------
% Function: y = sigmoid(x)
% Purpose : given x, generate y belongs to (0,1)
% Input   : x   --- input x belongs to R	
% Output  : y   --- output y belongs to (0,1)
% -------------- C. LIU & M. ZHAO -----------------

%
function y = sigmoid(x)
    
    y = 1 ./ (1 + exp(-x));

end

