function [output] = inner_product_forward(input, layer, param)

d = size(input.data, 1);
k = size(input.data, 2); % batch size
n = size(param.w, 2);

% Replace the following line with your implementation.
% the output of inner_product_forward should contain height, width,
% channel, batch_size and data
output.height = n;
output.width = 1;
output.channel = 1;
output.batch_size = k;
% data of output is given by: w * x + b
w = transpose(param.w);
x = input.data;
b = transpose(param.b);
output.data = w * x + b;

end
