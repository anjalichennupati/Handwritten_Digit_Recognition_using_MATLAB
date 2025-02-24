% Load MNIST dataset
Images = loadData("E:\2nd Year\4th Semester\Maths\implementation\MNIST\t10k-images.idx3-ubyte");
Labels = loadLabels("E:\2nd Year\4th Semester\Maths\implementation\MNIST\t10k-labels.idx1-ubyte");

% Display images
% Display first 25 training images
figure;
for i = 1:25
    subplot(5,5,i);
    imshow(reshape(Images(:,i),[28 28]), []);
    title(Labels(i));
end