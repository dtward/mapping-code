% test

clear all;
close all;

% some data
X = [0 0;
    1 0;
    0 1;
    1 1;
    0.5 0.5;
    0.5 0;
    0 0.5;
    0.5 1;
    1 0.5];
rng(10)
Y = X + (rand(size(X)) - 0.5)*0.5;

nT = 10;
epsilon = 0.00001;
sigmaV = 0.5;
sigmaY = 0.01;
nIter = 500;
[X1, Pt] = lddmmLandmark(X, Y, sigmaV, sigmaY, nT,epsilon, nIter);


% make a visualization
x = linspace(-0.5,1.5,20);
y = linspace(-0.5,1.5,20);
[xx,yy] = meshgrid(x,y);
Z = [xx(:),yy(:)];
Zt = flowPointsFromLandmarkPtSigma(X,Pt,sigmaV,Z);
xx1 = reshape(Zt(:,1,end),size(xx));
yy1 = reshape(Zt(:,2,end),size(xx));
figure;
scatter(X(:,1),X(:,2),'c','f')
hold on;
scatter(X1(:,1),X1(:,2),'b','f')
scatter(Y(:,1),Y(:,2),'r')
surf(xx1,yy1,xx1*0-1,'facecolor','none','linewidth',2)
legend('template','deformed','target','location','best')
axis image
axis off

% try the same test
[X1, Pt] = lddmmInitialMomentumLandmark(X, Y, sigmaV, sigmaY, nT,epsilon, nIter);
