function [output] = pooling_layer_forward(input, layer)

    h_in = input.height;
    w_in = input.width;
    c = input.channel;
    batch_size = input.batch_size;
    k = layer.k;
    pad = layer.pad;
    stride = layer.stride;
    
    h_out = (h_in + 2*pad - k) / stride + 1;
    w_out = (w_in + 2*pad - k) / stride + 1;
    
    
    output.height = h_out;
    output.width = w_out;
    output.channel = c;
    output.batch_size = batch_size;

    % Replace the following line with your implementation.
    output.data = zeros([h_out, w_out, c, batch_size]);
    for batch_idx = 1:batch_size
        % get the data for the current image in the batch
        current_input = input.data(:, batch_idx);
        % reshape the input image
        current_input = reshape(current_input, [h_in, w_in, c]);
        % pad the current image
        current_input = padarray(current_input, [pad pad]);
        % loop over the image to do the max pooling
        for r_idx = 1:h_out
            for c_idx = 1:w_out
                for channel_idx = 1:c
                    % compute the area for doing the pooling at the location
                    pooling_area = current_input(1+(r_idx-1)*stride:k+(r_idx-1)*stride, 1+(c_idx-1)*stride:k+(c_idx-1)*stride, channel);
                    % do max pooling on the area
                    output.data(r_idx, c_idx, channel_idx, batch_idx) = max(pooling_area, [], 'all');
                end
            end
        end
    end
    % reshape back the output
    output.data = reshape(output.data, [h_out * w_out * c], batch_size);
end

