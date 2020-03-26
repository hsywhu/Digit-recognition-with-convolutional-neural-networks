addpath('../matlab');

load lenet.mat
layers = get_lenet();

%% image 1
% turn image into binary image
image1_origin = imread('../images/image1.JPG');
image1 = 255 - rgb2gray(image1_origin);
image1 = im2double(image1);
threshold = 0.7;
image1_binary = binary(image1, threshold);

% find bounding boxes
connected = bwconncomp(image1_binary);
char_locs = connected.PixelIdxList;
[bb_top_left, bb_bot_right] = extract_boundingboxes(char_locs, size(image1_binary));
visualize_boundingboxes(image1_origin, bb_top_left, bb_bot_right);

% recognize the digit
model_input = generate_model_input(image1_binary, bb_top_left, bb_bot_right, 6, 'bicubic', 'false');
[output, P] = convnet_forward(params, layers, model_input);
 [~, argmaxP] = max(P);
 fig = figure;
for i = 1:size(bb_top_left, 1)
    input_img = transpose(reshape(model_input(:, i), [28, 28]));
    subplot(2, 5, i);
    imshow(input_img);
    title( sprintf('%g', argmaxP(:, i)-1) )
end
filename = '../results/testimage1.png';
frame = getframe(fig);
imwrite(frame2im(frame), filename);

% image 2
% turn image into binary image
image2_origin = imread('../images/image2.JPG');
image2 = 255 - rgb2gray(image2_origin);
image2 = im2double(image2);
threshold = 0.55;
image2_binary = binary(image2, threshold);

% find bounding boxes
connected = bwconncomp(image2_binary);
char_locs = connected.PixelIdxList;
[bb_top_left, bb_bot_right] = extract_boundingboxes(char_locs, size(image2_binary));
visualize_boundingboxes(image2_origin, bb_top_left, bb_bot_right);

% recognize the digit
model_input = generate_model_input(image2_binary, bb_top_left, bb_bot_right, 4, 'bilinear', 'true');
[output, P] = convnet_forward(params, layers, model_input);
 [~, argmaxP] = max(P);
 fig = figure;
for i = 1:size(bb_top_left, 1)
    input_img = transpose(reshape(model_input(:, i), [28, 28]));
    subplot(2, 5, i);
    imshow(input_img);
    title( sprintf('%g', argmaxP(:, i)-1) )
end
filename = '../results/testimage2.png';
frame = getframe(fig);
imwrite(frame2im(frame), filename);

%% image 3
% turn image into binary image
image3_origin = imread('../images/image3.png');
image3 = 255 - rgb2gray(image3_origin);
image3 = im2double(image3);
threshold = 0.8;
image3_binary = binary(image3, threshold);

% find bounding boxes
connected = bwconncomp(image3_binary);
char_locs = connected.PixelIdxList;
[bb_top_left, bb_bot_right] = extract_boundingboxes(char_locs, size(image3_binary));
visualize_boundingboxes(image3_origin, bb_top_left, bb_bot_right);

% recognize the digit
model_input = generate_model_input(image3_binary, bb_top_left, bb_bot_right, 4, 'bilinear', 'false');
[output, P] = convnet_forward(params, layers, model_input);
 [~, argmaxP] = max(P);
 fig = figure;
for i = 1:size(bb_top_left, 1)
    input_img = transpose(reshape(model_input(:, i), [28, 28]));
    subplot(1, 5, i);
    imshow(input_img);
    title( sprintf('%g', argmaxP(:, i)-1) )
end
filename = '../results/testimage3.png';
frame = getframe(fig);
imwrite(frame2im(frame), filename);

%% image 4
% turn image into binary image
image4_origin = imread('../images/image4.jpg');
image4 = 255 - rgb2gray(image4_origin);
image4 = im2double(image4);
threshold = 0.4;
image4_binary = binary(image4, threshold);

% find bounding boxes
connected = bwconncomp(image4_binary);
char_locs = connected.PixelIdxList;
[bb_top_left, bb_bot_right] = extract_boundingboxes(char_locs, size(image4_binary));
visualize_boundingboxes(image4_origin, bb_top_left, bb_bot_right);

% recognize the digit
model_input = generate_model_input(image4_binary, bb_top_left, bb_bot_right, 4, 'bilinear', 'false');
[output, P] = convnet_forward(params, layers, model_input);
 [~, argmaxP] = max(P);
 fig = figure;
for i = 1:size(bb_top_left, 1)
    input_img = transpose(reshape(model_input(:, i), [28, 28]));
    subplot(6, 10, i);
    imshow(input_img);
    title( sprintf('%g', argmaxP(:, i)-1) )
end
filename = '../results/testimage4.png';
frame = getframe(fig);
imwrite(frame2im(frame), filename);

%% functions
function [model_input] = generate_model_input(img, bb_top_left, bb_bot_right, pad_size, resize_method, double_binary)
    char_num = size(bb_top_left, 1);
    model_input = zeros(784, 100);
    for i = 1:char_num
        row_start = bb_top_left(i, 2);
        row_end = bb_bot_right(i, 2);
        col_start = bb_top_left(i, 1);
        col_end = bb_bot_right(i, 1);
        cropped = img(row_start:row_end, col_start:col_end);
        % pad the cropped image to a square img
        if size(cropped, 1) > size(cropped, 2)
            to_pad = round((size(cropped, 1) - size(cropped, 2))/2);
            cropped = padarray(cropped, [0 to_pad], 0, 'both');
        else
            to_pad = round((size(cropped, 2) - size(cropped, 1))/2);
            cropped = padarray(cropped, [to_pad 0], 0, 'both');
        end
        % resize to 26*26 then pad to 28*28
        cropped = imresize(cropped, [28-pad_size*2, 28-pad_size*2], 'method', resize_method);
        if strcmp(double_binary, 'true')
        cropped(cropped > 0.1) = 1;
        end
        cropped = padarray(cropped, [pad_size pad_size], 0, 'both');
        cropped = transpose(cropped);
        model_input(:, i) = reshape(cropped, [784, 1]);
    end
end

function [] = visualize_boundingboxes(img_origin, bb_top_left, bb_bot_right)
    figure;
    imshow(img_origin);
    hold on;
    for i = 1:size(bb_top_left, 1)
        x_len = (bb_bot_right(i, 1)-bb_top_left(i, 1));
        y_len = (bb_bot_right(i, 2)-bb_top_left(i, 2));
        rectangle('Position', [bb_top_left(i, 1), bb_top_left(i, 2), x_len, y_len], 'EdgeColor', [1, 0, 0], 'LineWidth', 2);
    end
    hold off;
end

function [img_binary] =  binary(img, threshold)
    % input img should be double type from 0 to 1
    img_binary = img;
    img_binary(img >= threshold) = 1;
    img_binary(img < threshold) = 0;
end

function [bb_top_left, bb_bot_right] = extract_boundingboxes(locs, img_size)
    char_num = size(locs, 2);
    bb_top_left = zeros(char_num, 2);
    bb_bot_right = zeros(char_num, 2);
    for i=1:char_num
        single_char_locs = locs{i};
        [rows, cols] = ind2sub(img_size, single_char_locs);
        bb_top_left(i, :) = [min(cols), min(rows)];
        bb_bot_right(i, :) = [max(cols), max(rows)];
    end
end