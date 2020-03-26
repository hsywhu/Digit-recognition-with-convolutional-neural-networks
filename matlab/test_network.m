%% Network defintion
layers = get_lenet();

%% Loading data
fullset = false;
[xtrain, ytrain, xvalidate, yvalidate, xtest, ytest] = load_mnist(fullset);
% load the trained weights
load lenet.mat

%% Testing the network
% Modify the code to get the confusion matrix

% init the confusion matrix
C = zeros(10, 10);

% loop over the test images to fill in the confusion matrix
for i=1:100:size(xtest, 2)
    [output, P] = convnet_forward(params, layers, xtest(:, i:i+99));
    
    % the prediction is the class with the biggest probability
    [~, argmaxP] = max(P);
    
    % put the result to confusion matrix
    for j = 1:size(argmaxP, 2)
        test_idx = i+j-1;
        C(ytest(test_idx), argmaxP(j)) = C(ytest(test_idx), argmaxP(j)) + 1;
    end
end

disp(C);