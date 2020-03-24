function [output] = conv_layer_forward(input, layer, param)
% Conv layer forward
% input: struct with input data
% layer: convolution layer struct
% param: weights for the convolution layer

% output: 

h_in = input.height;
w_in = input.width;
c = input.channel;
batch_size = input.batch_size;
k = layer.k;
pad = layer.pad;
stride = layer.stride;
num = layer.num;
% resolve output shape
h_out = (h_in + 2*pad - k) / stride + 1;
w_out = (w_in + 2*pad - k) / stride + 1;

assert(h_out == floor(h_out), 'h_out is not integer')
assert(w_out == floor(w_out), 'w_out is not integer')
input_n.height = h_in;
input_n.width = w_in;
input_n.channel = c;

%% Fill in the code
% Iterate over the each image in the batch, compute response,
% Fill in the output datastructure with data, and the shape. 

% fill in the information part first
output.height = h_out;
output.width = w_out;
output.channel = num;
output.batch_size = batch_size;

% init the output datastructure
output.data = zeros([h_out, w_out, num, batch_size]);

% loop over the images to do the convolution
for batch_idx = 1:batch_size
    % get the data for current image 
    current_input = input.data(:, batch_idx);
    % reshape the input image
    current_input = reshape(current_input, [h_in, w_in, c]);
    % pad the current image
    current_input = padarray(current_input, [pad pad]);
    % loop over the image to do the max pooling
    for r_idx = 1:h_out
        for c_idx = 1:w_out
            for filter_idx = 1:num
                % compute the area for doing the convolution at the location
                conv_area = current_input(1+(r_idx-1)*stride:k+(r_idx-1)*stride, 1+(c_idx-1)*stride:k+(c_idx-1)*stride, :);
                % extract the filter from param
                current_filter = param.w(:, filter_idx);
                current_filter = reshape(current_filter, [k k c]);
                % extract the bias from param
                current_bias = param.b(:, filter_idx);
                % do convolution on the selected area
                conv_res = sum(conv_area .* current_filter, 'all') + current_bias;
                output.data(r_idx, c_idx, filter_idx, batch_idx) = conv_res;
            end
        end
    end
end
output.data = reshape(output.data, [h_out * w_out * num, batch_size]);
end

