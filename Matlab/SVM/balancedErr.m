function [matrix, BER] = balancedErr(nnPred, LLTrue)
    
    samples  = size(LLTrue,1);
    matrix   = zeros(4);
            
    for i = 1:samples        
        idx = find(max(nnPred(i,:)) == (nnPred(i,:))); 
        if idx == LLTrue(i)            
            switch idx
                case {1}
                    matrix(1,1) = matrix(1,1) + 1;
                case {2}
                    matrix(2,2) = matrix(2,2) + 1;
                case {3}
                    matrix(3,3) = matrix(3,3) + 1;
                case {4}
                    matrix(4,4) = matrix(4,4) + 1;
            end
            
        else
            switch LLTrue(i)
                case {1}
                        if idx == 2
                            matrix(1,2) = matrix(1,2) + 1;
                        end
                        if idx == 3
                            matrix(1,3) = matrix(1,3) + 1;
                        end
                        if idx == 4
                            matrix(1,4) = matrix(1,4) + 1;
                        end
                case {2}
                        if idx == 1
                            matrix(2,1) = matrix(2,1) + 1;
                        end
                        if idx == 3
                            matrix(2,3) = matrix(2,3) + 1;
                        end
                        if idx == 4
                            matrix(2,4) = matrix(2,4) + 1;
                        end                    
                case {3}
                        if idx == 1
                            matrix(3,1) = matrix(3,1) + 1;
                        end
                        if idx == 2
                            matrix(3,2) = matrix(3,2) + 1;
                        end
                        if idx == 4
                            matrix(3,4) = matrix(3,4) + 1;
                        end
                 case {4}
                        if idx == 1
                            matrix(4,1) = matrix(4,1) + 1;
                        end
                        if idx == 2
                            matrix(4,2) = matrix(4,2) + 1;
                        end
                        if idx == 3
                            matrix(4,3) = matrix(4,3) + 1;
                        end
            end                      
        end                                     
    end
   
    for i = 1:4
          e(i) = matrix(i,i)/(sum(matrix(i,:)));
    end
    
    BER = mean(e);
                
            
         
    
    
    