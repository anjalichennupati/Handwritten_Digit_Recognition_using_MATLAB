clear all

load('MnistConv.mat')

% Load the input image from your computers
imageFile = "C:\Users\Anjali\Downloads\Untitled.png"; % Replace with the actual path to your image file
x = imread(imageFile);

% Convert the image to grayscale if necessary
if size(x, 3) > 1
    x = rgb2gray(x);
end

% Resize the image to match the input size of the CNN (28x28)
x = imresize(x, [28, 28]);

% Normalize the pixel values to the range [0, 1]
x = double(x) / 255;

y1 = Conv(x, W1);                 % Convolution,  20x20x20
y2 = ReLU(y1);                    %
y3 = Pool(y2);                    % Pool,         10x10x20
y4 = reshape(y3, [], 1);          %                   2000  
v5 = W5 * y4;                     % ReLU,              360
y5 = ReLU(v5);                    %
v  = Wo * y5;                     % Softmax,            10
y  = Softmax(v);                  %
  
figure;
imshow(x);
title('Input Image')

convFilters = zeros(9 * 9, 20);
for i = 1:20
  filter            = W1(:, :, i);
  convFilters(:, i) = filter(:);
end
figure
display_network(convFilters);
title('Convolution Filters')

fList = zeros(20 * 20, 20);
for i = 1:20
  feature     = y1(:, :, i);
  fList(:, i) = feature(:);
end
figure
display_network(fList);
title('Features [Convolution]')

fList = zeros(20 * 20, 20);
for i = 1:20
  feature     = y2(:, :, i);
  fList(:, i) = feature(:);
end
figure
display_network(fList);
title('Features [Convolution + ReLU]')

fList = zeros(10 * 10, 20);
for i = 1:20
  feature     = y3(:, :, i);
  fList(:, i) = feature(:);
end
figure
display_network(fList);
title('Features [Convolution + ReLU + MeanPool]')

% Display the predicted class
[~, prediction] = max(y);
fprintf('Predicted class: %d\n', prediction);
