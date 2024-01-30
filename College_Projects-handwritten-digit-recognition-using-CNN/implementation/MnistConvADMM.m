function [W1, W5, Wo] = MnistConvADMM(W1, W5, Wo, X, D, lambda, rho, maxIterations)
%
% Inputs:
%   W1: Convolutional layer weights
%   W5: Fully connected layer weights
%   Wo: Output layer weights
%   X: Input images
%   D: Ground truth labels
%   lambda: ADMM penalty parameter
%   rho: ADMM dual update step size
%   maxIterations: Maximum number of ADMM iterations
%
% Outputs:
%   W1: Updated Convolutional layer weights
%   W5: Updated Fully connected layer weights
%   Wo: Updated Output layer weights

alpha = 0.01;
beta = 0.95;

momentum1 = zeros(size(W1));
momentum5 = zeros(size(W5));
momentumo = zeros(size(Wo));

N = length(D);

bsize = 100;  
blist = 1:bsize:(N-bsize+1);

% One epoch loop
for batch = 1:length(blist)
    dW1 = zeros(size(W1));
    dW5 = zeros(size(W5));
    dWo = zeros(size(Wo));
    U1 = zeros(size(W1));
    U5 = zeros(size(W5));
    Uo = zeros(size(Wo));

    % Mini-batch loop
    begin = blist(batch);
    for k = begin:begin+bsize-1
        % Forward pass = inference
        x  = X(:, :, k);
        y1 = Conv(x, W1);
        y2 = ReLU(y1);
        y3 = Pool(y2);
        y4 = reshape(y3, [], 1);
        v5 = W5*y4;
        y5 = ReLU(v5);
        v  = Wo*y5;
        y  = Softmax(v);

        d = zeros(10, 1);
        d(sub2ind(size(d), D(k), 1)) = 1;

        e      = d - y;
        delta  = e;

        e5     = Wo' * delta;
        delta5 = (y5 > 0) .* e5;

        e4     = W5' * delta5;

        e3     = reshape(e4, size(y3));

        e2 = zeros(size(y2));           
        W3 = ones(size(y2)) / (2*2);
        for c = 1:20
            e2(:, :, c) = kron(e3(:, :, c), ones([2 2])) .* W3(:, :, c);
        end

        delta2 = (y2 > 0) .* e2;

        delta1_x = zeros(size(W1));
        for c = 1:20
            delta1_x(:, :, c) = conv2(x(:, :), rot90(delta2(:, :, c), 2), 'valid');
        end

        dW1 = dW1 + delta1_x; 
        dW5 = dW5 + delta5*y4';    
        dWo = dWo + delta *y5';
        
        % ADMM update
        Z1 = W1 - U1;
        Z5 = W5 - U5;
        Zo = Wo - Uo;
        
        W1 = shrinkage(Z1, lambda/rho);
        W5 = shrinkage(Z5, lambda/rho);
        Wo = shrinkage(Zo, lambda/rho);
        
        U1 = U1 + alpha*(W1 - Z1);
        U5 = U5 + alpha*(W5 - Z5);
        Uo = Uo + alpha*(Wo - Zo);
    end
    
    % Update weights with momentum
    momentum1 = alpha*dW1 + beta*momentum1;
    momentum5 = alpha*dW5 + beta*momentum5;
    momentumo = alpha*dWo + beta*momentumo;
    
    W1 = W1 + momentum1;
    W5 = W5 + momentum5;
    Wo = Wo + momentumo;
    
    % Learning rate decay
    alpha = alpha * 0.99;
end

end

function Z = shrinkage(X, kappa)
    Z = max(0, X - kappa) - max(0, -X - kappa);
end
