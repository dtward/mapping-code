% deform image I to match J
% regularization energy is \|Lv\|^2_{L_2}/2, L = Id - alpha^2 Laplacian,
% matching energy is \|I(phiinv) - J\|^2_{L_2}/2/sigma^2 for phiinv(x) = x - vx 
% optimization using gradient descent with stepsize epsilon for nIter steps
% energy gradient is -(I(x - v) - J(x))nabla(I)(x - v)
function [ID,vx,vy] = splineImage(I,J,alpha,sigma,epsilon,nIter)
% initialize velocity field
vx = zeros(size(I));
vy = zeros(size(I));
for i = 1 : nIter
    % deform image
    ID = applyVToImage(I,vx,vy);
    % energy (cost function)
    E = sum(sum(sum(applyPowerOfA(vx,alpha,1).*vx + applyPowerOfA(vy,alpha,1).*vy)))/2 + sum(sum((ID-J).^2))/2/sigma^2;
    fprintf('Iter %d of %d, energy is %g\n',i,nIter,E);    
    % gradient of I
    [gradIx,gradIy] = gradient(I);
    % deform gradient
    gradIxD = applyVToImage(gradIx,vx,vy);
    gradIyD = applyVToImage(gradIy,vx,vy);
    % gradient of matching term
    gradMatchx = applyPowerOfA( -(ID - J).*gradIxD/sigma^2, alpha, -1);
    gradMatchy = applyPowerOfA( -(ID - J).*gradIyD/sigma^2, alpha, -1);
    % energy gradient
    gradx = vx + gradMatchx;
    grady = vy + gradMatchy;
    % update velocity
    vx = vx - gradx*epsilon;
    vy = vy - grady*epsilon;    
end

function ID = applyVToImage(I,vx,vy)
% deform image by composing with x-v (interpolating at specified points)
%[X,Y] = meshgrid(0:size(I,2)-1,0:size(I,1)-1);
[X,Y] = meshgrid(1:size(I,2),1:size(I,1));
ID = interp2(I,X-vx,Y-vy,'linear',0);
% ID = interp2(I,X-vx,Y-vy,'nearest',0);

function p = applyPowerOfA(v,alpha,power)
% A = (1 - alpha^2 Laplacian)^2, use discrete Laplacian in Fourier domain
[FX,FY] = meshgrid((0:size(v,2)-1)/size(v,2),(0:size(v,1)-1)/size(v,1));
Apow = ( 1 - alpha^2*2*(cos(2*pi*FX) + cos(2*pi*FY) - 2 ) ).^(2*power);
p = ifft2( bsxfun(@times, fft2(v), Apow), 'symmetric');