function [param_grad, input_od] = inner_product_backward(output, input, layer, param)

% Replace the following lines with your implementation.
% init b and w
param_grad.b = zeros(size(param.b));
param_grad.w = zeros(size(param.w));

% input_od can be directly calculated
input_od = param.w * output.diff;

for batch_idx = 1:input.batch_size
    % backward on bias
    current_diff = transpose(output.diff(:, batch_idx));
    param_grad.b = param_grad.b + current_diff;
    
    % backward on weight
    current_input = input.data(:, batch_idx);
    param_grad.w = param_grad.w + current_input * current_diff;
end

end
