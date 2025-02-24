function y = Pool(x)
% x - input tensor
% y - output tensor
[xrow, xcol, numFilters] = size(x);
y = zeros(xrow/2, xcol/2, numFilters);

% max pooling for each fllter in x
for k = 1:numFilters
    filter = ones(2)/(2*2);
    image = conv2(x(:,:,k), filter, 'valid');
    % downsampling - selecting alternate elements in row and column
    y(:,:,k) = image (1:2:end, 1:2:end);
end
end