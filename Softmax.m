function soft_out = Softmax(inp)

    % Softmax - applied to produce a prob dist over multiple classes in NN
    % inp - input vector

    % find exp values 
    soft_out = exp(inp);
    
    % normalising soft_out
    soft_out = soft_out./sum(soft_out);
end