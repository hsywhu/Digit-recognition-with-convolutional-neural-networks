layers = get_lenet();
load lenet.mat
% load data
% Change the following value to true to load the entire dataset.
fullset = false;
[xtrain, ytrain, xvalidate, yvalidate, xtest, ytest] = load_mnist(fullset);
xtrain = [xtrain, xvalidate];
ytrain = [ytrain, yvalidate];
m_train = size(xtrain, 2);
batch_size = 64;
 
 
layers{1}.batch_size = 1;
img = xtest(:, 1);
img = reshape(img, 28, 28);
imshow(img');
figure;
 
%[cp, ~, output] = conv_net_output(params, layers, xtest(:, 1), ytest(:, 1));
output = convnet_forward(params, layers, xtest(:, 1));
output_1 = reshape(output{1}.data, 28, 28);
% Fill in your code here to plot the features.

% layer 2
output_2 = output{2}.data;
% output at layer 2 is 24 * 24
output_2 = reshape(output_2, [24, 24, 20]);
for i = 1:20
    subplot(4, 5, i);
    output_2_tmp = transpose(output_2(:, :, i));
    imshow(output_2_tmp);
end

% layer 3
output_3 = output{3}.data;
% output at layer 3 is also 24 * 24
output_3 = reshape(output_3, [24, 24, 20]);
figure;
for i = 1:20
    subplot(4, 5, i);
    output_3_tmp = transpose(output_3(:, :, i));
    imshow(output_3_tmp);
end

