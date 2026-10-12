%% SC4061/CE4003/CZ4003 Computer Vision - Lab 1
clear; close all; clc;

%% 2.1 Contrast Stretching
% a. Read the image and convert to grayscale
Pc = imread('mrt-train.jpg');
whos Pc
P = rgb2gray(Pc);

% b. View the original image
figure('Name', '2.1 Contrast Stretching');
subplot(1,2,1); imshow(P); title('Original P');

% c. Minimum and maximum intensities
fprintf('2.1 Original: min = %d, max = %d\n', min(P(:)), max(P(:)));

% d. Contrast stretching: shift min to 0, then scale max to 255
P2 = imsubtract(P, double(min(P(:))));             % 13..204 -> 0..191
P2 = immultiply(P2, 255 / double(max(P2(:))));     % 0..191 -> 0..255
fprintf('2.1 Stretched: min = %d, max = %d\n', min(P2(:)), max(P2(:)));

% e. Display the stretched image
subplot(1,2,2); imshow(P2); title('Contrast stretched P2');

%% 2.2 Histogram Equalization
P = rgb2gray(imread('mrt-train.jpg'));

% a. Histograms of P with 10 and 256 bins
figure('Name', '2.2 Histogram Equalization');
subplot(3,2,1); imhist(P, 10);  title('P, 10 bins');
subplot(3,2,2); imhist(P, 256); title('P, 256 bins');

% b. Histogram equalization
P3 = histeq(P, 255);
subplot(3,2,3); imhist(P3, 10);  title('P3 = histeq(P), 10 bins');
subplot(3,2,4); imhist(P3, 256); title('P3 = histeq(P), 256 bins');

% c. Equalize again
P4 = histeq(P3, 255);
subplot(3,2,5); imhist(P4, 10);  title('P4 = histeq(P3), 10 bins');
subplot(3,2,6); imhist(P4, 256); title('P4 = histeq(P3), 256 bins');
% How much did the second equalization change?
fprintf('2.2 Pixels changed by 2nd histeq: %d of %d (%.2f%%)\n', ...
        nnz(P3 ~= P4), numel(P3), 100 * nnz(P3 ~= P4) / numel(P3));
fprintf('2.2 Largest change: %d gray level(s)\n', ...
        max(abs(double(P3(:)) - double(P4(:)))));
fprintf('2.2 Distinct gray levels: P = %d, P3 = %d, P4 = %d\n', ...
        numel(unique(P)), numel(unique(P3)), numel(unique(P4)));

figure('Name', '2.2 Equalized images');
subplot(1,3,1); imshow(P);  title('P');
subplot(1,3,2); imshow(P3); title('P3 = histeq(P)');
subplot(1,3,3); imshow(P4); title('P4 = histeq(P3)');

%% 2.3 Linear Spatial Filtering
% a. Gaussian filters, 5x5, sigma = 1 and sigma = 2
[xx, yy] = meshgrid(-2:2, -2:2);      % x, y coordinates from -2 to 2

sigma1 = 1.0;
h1 = exp(-(xx.^2 + yy.^2) / (2 * sigma1^2)) / (2 * pi * sigma1^2);
h1 = h1 / sum(h1(:));                 % normalize so elements sum to 1

sigma2 = 2.0;
h2 = exp(-(xx.^2 + yy.^2) / (2 * sigma2^2)) / (2 * pi * sigma2^2);
h2 = h2 / sum(h2(:));                 % normalize so elements sum to 1

fprintf('2.3 sum(h1) = %.4f, sum(h2) = %.4f\n', sum(h1(:)), sum(h2(:)));

figure('Name', '2.3 Gaussian filters');
subplot(1,2,1); mesh(h1); title('Gaussian 5x5, \sigma = 1');
subplot(1,2,2); mesh(h2); title('Gaussian 5x5, \sigma = 2');

% b, c. Image with additive Gaussian noise
Pgn = read_gray('lib-gn.jpg');
Pgn_h1 = uint8(conv2(double(Pgn), h1, 'same'));
Pgn_h2 = uint8(conv2(double(Pgn), h2, 'same'));

figure('Name', '2.3 Gaussian filtering on Gaussian noise');
subplot(1,3,1); imshow(Pgn);    title('lib-gn (Gaussian noise)');
subplot(1,3,2); imshow(Pgn_h1); title('Gaussian \sigma = 1');
subplot(1,3,3); imshow(Pgn_h2); title('Gaussian \sigma = 2');

% d, e. Image with additive speckle noise
Psp = read_gray('lib-sp.jpg');
Psp_h1 = uint8(conv2(double(Psp), h1, 'same'));
Psp_h2 = uint8(conv2(double(Psp), h2, 'same'));

figure('Name', '2.3 Gaussian filtering on speckle noise');
subplot(1,3,1); imshow(Psp);    title('lib-sp (speckle noise)');
subplot(1,3,2); imshow(Psp_h1); title('Gaussian \sigma = 1');
subplot(1,3,3); imshow(Psp_h2); title('Gaussian \sigma = 2');

%% 2.4 Median Filtering
Pgn = read_gray('lib-gn.jpg');
Psp = read_gray('lib-sp.jpg');
Pgn_m3 = medfilt2(Pgn, [3 3]);
Pgn_m5 = medfilt2(Pgn, [5 5]);
Psp_m3 = medfilt2(Psp, [3 3]);
Psp_m5 = medfilt2(Psp, [5 5]);

figure('Name', '2.4 Median filtering on Gaussian noise');
subplot(1,3,1); imshow(Pgn);    title('lib-gn (Gaussian noise)');
subplot(1,3,2); imshow(Pgn_m3); title('Median 3x3');
subplot(1,3,3); imshow(Pgn_m5); title('Median 5x5');

figure('Name', '2.4 Median filtering on speckle noise');
subplot(1,3,1); imshow(Psp);    title('lib-sp (speckle noise)');
subplot(1,3,2); imshow(Psp_m3); title('Median 3x3');
subplot(1,3,3); imshow(Psp_m5); title('Median 5x5');

%% 2.5 Suppressing Noise Interference Patterns
% a. Read and display the image
Pi = read_gray('pck-int.jpg');
figure('Name', '2.5 Interference: original');
imshow(Pi); title('pck-int');

% b. Fourier transform and power spectrum
F = fft2(double(Pi));
S = abs(F).^2;
figure('Name', '2.5 Power spectrum (shifted)');
imagesc(fftshift(S.^0.1)); colormap('default'); colorbar;
title('fftshift(S)^{0.1}');

% c. Power spectrum without fftshift, and the peak locations.
% Click the two bright peaks (set USE_GINPUT = false to find them automatically).
figure('Name', '2.5 Power spectrum (unshifted)');
imagesc(S.^0.1); colormap('default'); colorbar;
title('S^{0.1} (no fftshift)');

USE_GINPUT = true;
if USE_GINPUT
    [px, py] = ginput(2);
    peaks = round([px py]);
else
    peaks = find_peaks(S, 2, 10);
end
fprintf('2.5 Interference peaks (x, y): (%d, %d) and (%d, %d)\n', peaks');

% d. Zero the 5x5 neighbourhood at each peak in F
F2 = F;
for k = 1:size(peaks, 1)
    F2 = zero_block(F2, peaks(k,1), peaks(k,2), 2);
end
    S2 = abs(F2).^2;
figure('Name', '2.5 Power spectrum after removing peaks');
imagesc(fftshift(S2.^0.1)); colormap('default'); colorbar;
title('Peaks set to zero (5x5)');

% e. Inverse Fourier transform
Pi2 = real(ifft2(F2));

% Improvement: also zero the horizontal and vertical lines through each peak,
% which still carry part of the interference energy
F3 = F2;
[M, N] = size(F3);
for k = 1:size(peaks, 1)
    x = peaks(k,1); y = peaks(k,2);
    F3(y, :) = 0;
    F3(:, x) = 0;
end
F3(1,1) = F(1,1);   % keep the DC component (average brightness)
Pi3 = real(ifft2(F3));

figure('Name', '2.5 Interference removal results');
subplot(1,3,1); imshow(Pi);                 title('Original');
subplot(1,3,2); imshow(uint8(Pi2));         title('5x5 peaks removed');
subplot(1,3,3); imshow(contrast_stretch(Pi3)); title('Peak rows/columns removed');

% f. Primate behind a fence
Pp = read_gray('primate-caged.jpg');
Fp = fft2(double(Pp));
Fs = fftshift(Fp);                              % centred spectrum

% The fence is a 2D lattice, so its peaks lie on 3 parallel lines in the
% spectrum (one through the centre, one on each side). Remove a thin band
% along each line, keeping the centre (the primate's overall shape).
% Peak offsets measured from the centred spectrum (relative to the centre):
d1 = [115 -49];     % far harmonic of the main fence peak -> line direction
p2 = [9 -15];       % strongest peak on the neighbouring line
[M, N] = size(Pp);
[xx, yy] = meshgrid(1:N, 1:M);
c = [N/2 + 1, M/2 + 1];                         % centre (zero frequency)
u = d1 / norm(d1);                              % unit direction of the lines
w = 2.5;                                        % band half-width (pixels)
keep_centre = 1 ./ (1 + (hypot(xx - c(1), yy - c(2)) / 7).^-8);  % 0 at centre
H = ones(M, N);
for o = {c, c + p2, c - p2}                     % one point on each line
    q = o{1};
    dist = abs(-(xx - q(1)) * u(2) + (yy - q(2)) * u(1));   % distance to line
    H = H .* (1 - exp(-dist.^2 / (2 * w^2)) .* keep_centre);
end
Fs2 = Fs .* H;
Pp2 = real(ifft2(ifftshift(Fs2)));

figure('Name', '2.5f Primate power spectrum');
subplot(1,2,1); imagesc((abs(Fs).^2).^0.1);  colormap('default'); axis image;
title('Before');
subplot(1,2,2); imagesc((abs(Fs2).^2).^0.1); colormap('default'); axis image;
title('After band filter');

figure('Name', '2.5f Freeing the primate');
subplot(1,2,1); imshow(Pp);         title('Original');
subplot(1,2,2); imshow(uint8(Pp2)); title('Fence removed (band filter)');

%% 2.6 Undoing Perspective Distortion of Planar Surface
% a. Read and display the book image
P = imread('book.jpg');
figure('Name', '2.6 Book (original)');
imshow(P); title('book.jpg');

% b. Corners of the book in the order:
%    top-left, top-right, bottom-right, bottom-left
% Click the 4 corners in the order above (set USE_GINPUT_BOOK = false to use
% the saved corner values instead).
USE_GINPUT_BOOK = true;
if USE_GINPUT_BOOK
    [X, Y] = ginput(4);
else
    X = [143; 308; 257;   4];
    Y = [ 28;  47; 216; 159];
end
% Desired corners of an A4 page (210 mm x 297 mm), same order
x = [0; 210; 210;   0];
y = [0;   0; 297; 297];

corner_names = {'Top-left', 'Top-right', 'Bottom-right', 'Bottom-left'};
fprintf('2.6 Corners:\n');
fprintf('  %-13s %-16s %s\n', 'Corner', 'Book (X, Y)', 'A4 target (x, y)');
for k = 1:4
    fprintf('  %-13s %-16s %s\n', corner_names{k}, ...
            sprintf('(%.0f, %.0f)', X(k), Y(k)), sprintf('(%d, %d)', x(k), y(k)));
end

% c. Set up A u = v from equation (*) and solve
A = zeros(8, 8);
v = zeros(8, 1);
for k = 1:4
    A(2*k-1, :) = [X(k) Y(k) 1 0 0 0 -x(k)*X(k) -x(k)*Y(k)];
    A(2*k,   :) = [0 0 0 X(k) Y(k) 1 -y(k)*X(k) -y(k)*Y(k)];
    v(2*k-1) = x(k);
    v(2*k)   = y(k);
end
u = A \ v;
U = reshape([u; 1], 3, 3)';
disp('2.6 Projective transformation matrix U:'); disp(U);

% Verify: transform the original corners
w = U * [X'; Y'; ones(1,4)];
w = w ./ (ones(3,1) * w(3,:));
disp('2.6 Transformed corners (should be the A4 corners):'); disp(w);

% d. Warp the image
T = maketform('projective', U');
P2 = imtransform(P, T, 'XData', [0 210], 'YData', [0 297]);

% e. Display the result
figure('Name', '2.6 Book (frontal view)');
imshow(P2); title('Warped to frontal view (1 pixel = 1 mm)');

% f. Find the big pink rectangle (the computer screen) by colour
hsv = rgb2hsv(P2);
H = hsv(:,:,1); Sat = hsv(:,:,2); Val = hsv(:,:,3);
pink = (H > 0.85 | H < 0.05) & Sat > 0.25 & Val > 0.5;
pink = imopen(pink, strel('square', 3));   % remove small specks
pink = bwareafilt(pink, 1);                % keep the largest pink region
stats = regionprops(pink, 'BoundingBox');
bbox = stats(1).BoundingBox;
fprintf('2.6f Pink screen bounding box [x y width height] = [%.1f %.1f %.1f %.1f]\n', bbox);

figure('Name', '2.6f Pink screen');
subplot(1,2,1); imshow(pink); title('Pink mask');
subplot(1,2,2); imshow(P2); hold on;
rectangle('Position', bbox, 'EdgeColor', 'g', 'LineWidth', 2);
title('Detected pink screen'); hold off;

%% 2.7 Code Two Perceptrons
% Example from the lecture slides (L1, Section 6.2):
% x(1) = [3 3 1]' in class c1, x(2) = [1 1 1]' in class c2,
% alpha = 1, w(1) = [0 0 0]'. The last element 1 is the added constant
% dimension, so w(3) is the bias.
Xp = [3 1;
      3 1;
      1 1];          % each column is one sample
rp = [1 -1];         % +1 = class c1, -1 = class c2

% Algorithm 1: update only when a sample is misclassified
%   1') x in c1 and w'x <= 0:  w = w + alpha*x
%   2') x in c2 and w'x >= 0:  w = w - alpha*x
%   3') otherwise:             w unchanged
% The samples are presented in turn, k = 1, 2, 3, ... until both are
% classified correctly in a row.
alpha1 = 1;
w1 = [0; 0; 0];
k = 1;
n_correct = 0;
fprintf('2.7 Algorithm 1 iterations:\n');
while n_correct < size(Xp, 2)
    i = mod(k - 1, size(Xp, 2)) + 1;      % which sample: 1, 2, 1, 2, ...
    x = Xp(:, i);
    if rp(i) == 1 && w1' * x <= 0
        w1 = w1 + alpha1 * x;
        n_correct = 0;
    elseif rp(i) == -1 && w1' * x >= 0
        w1 = w1 - alpha1 * x;
        n_correct = 0;
    else
        n_correct = n_correct + 1;
    end
    fprintf('  w(%d) = [%s]\n', k + 1, num2str(w1'));
    k = k + 1;
end
fprintf('2.7 Algorithm 1 result: w = [%s], boundary %g*x1 + %g*x2 + %g = 0\n', ...
        num2str(w1'), w1);

% Algorithm 2 (slide 6.3): gradient descent on E(w) = 1/2 (r - w'x)^2
%   w(k+1) = w(k) + alpha * (r(k) - w(k)'x(k)) * x(k)
% Every sample causes an update, even if classified correctly, so w keeps
% moving until w'x is close to the target r (+1 or -1).
alpha2 = 0.1;
n_iter = 100;
w2 = [0; 0; 0];
cost = zeros(1, n_iter);
for k = 1:n_iter
    i = mod(k - 1, size(Xp, 2)) + 1;
    x = Xp(:, i);
    w2 = w2 + alpha2 * (rp(i) - w2' * x) * x;
    cost(k) = mean(0.5 * (rp - w2' * Xp).^2);   % average cost over both samples
end
fprintf('2.7 Algorithm 2 result: w = [%s], final cost = %.4f\n', ...
        num2str(w2', 4), cost(end));
fprintf('2.7 Algorithm 2 outputs w''x = [%s] (targets [1 -1])\n', ...
        num2str(w2' * Xp, 4));

figure('Name', '2.7 Perceptrons');
subplot(1,3,1); plot_perceptron(Xp(1:2,:), rp, w1); title('Algorithm 1');
subplot(1,3,2); plot_perceptron(Xp(1:2,:), rp, w2); title('Algorithm 2');
subplot(1,3,3); plot(1:n_iter, cost, 'LineWidth', 1.5);
xlabel('Iterations'); ylabel('Cost'); title('Algorithm 2 cost'); grid on;

%% Save all open figures as PNG files in the 'output' folder
if ~isfolder('output')
    mkdir('output');
end
figs = findall(0, 'Type', 'figure');
for k = 1:numel(figs)
    if ~isvalid(figs(k)), continue; end         % skip figures already closed
    name = figs(k).Name;
    if isempty(name)
        name = sprintf('figure_%d', figs(k).Number);
    end
    name = regexprep(name, '[^\w\.\- ]', '_');   % make it a safe file name
    exportgraphics(figs(k), fullfile('output', [name '.png']), 'Resolution', 150);
end
fprintf('Saved %d figures to the output folder\n', numel(figs));

%% ===================== Helper functions =====================

function I = read_gray(fname)
% Read an image and convert to grayscale if it is RGB
I = imread(fname);
if ndims(I) == 3
    I = rgb2gray(I);
end
end

function out = contrast_stretch(I)
% Linearly stretch a double image to 0..255 and convert to uint8
I = double(I);
out = uint8(255 * (I - min(I(:))) / (max(I(:)) - min(I(:))));
end

function peaks = find_peaks(S, n, r0)
% Find the n strongest peaks of an (unshifted) power spectrum S, ignoring
% the low frequencies within radius r0 of the DC component.
% Returns [x y] coordinates in the unshifted spectrum.
[M, N] = size(S);
[xx, yy] = meshgrid(1:N, 1:M);
dx = min(xx - 1, N - xx + 1);     % distance from DC, with wrap-around
dy = min(yy - 1, M - yy + 1);
S(sqrt(dx.^2 + dy.^2) < r0) = 0;

peaks = zeros(n, 2);
for k = 1:n
    [~, idx] = max(S(:));
    [py, px] = ind2sub([M N], idx);
    peaks(k, :) = [px py];
    S = zero_block(S, px, py, 2);  % suppress so the next max is a new peak
end
end

function F = zero_block(F, x, y, r)
% Set the (2r+1)x(2r+1) neighbourhood centred at (x, y) to zero,
% wrapping around the edges (the spectrum is periodic)
[M, N] = size(F);
rows = mod((y - r : y + r) - 1, M) + 1;
cols = mod((x - r : x + r) - 1, N) + 1;
F(rows, cols) = 0;
end

function plot_perceptron(X, y, w)
% Plot the samples and the decision boundary w1*x1 + w2*x2 + w3 = 0
hold on;
plot(X(1, y == 1),  X(2, y == 1),  'bo', 'MarkerSize', 8, 'LineWidth', 2);
plot(X(1, y == -1), X(2, y == -1), 'rx', 'MarkerSize', 8, 'LineWidth', 2);
x1 = linspace(min(X(1,:)) - 1, max(X(1,:)) + 1, 100);
if w(2) ~= 0
    plot(x1, -(w(1) * x1 + w(3)) / w(2), 'g-', 'LineWidth', 2);
else
    xline(-w(3) / w(1), 'g-', 'LineWidth', 2);
end
legend('Class +1', 'Class -1', 'Decision boundary', 'Location', 'best');
xlabel('x_1'); ylabel('x_2'); grid on; axis equal;
hold off;
end
