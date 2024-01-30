function out = ReLU(conv)
    % ReLU - Rectified Linear unit - introduces non linearity into a neural
    % network
    % conv - complex valued variable
    % ReLU is applied to the real and imaginary part of conv separately

    out = max(0,real(conv)) + 1i*(max(0, imag(conv)));
end
