% test
X = [0 0;
    1 0;
    0 1;
    1 1;
    0.5 0.5];
rng(2)
Y = X + randn(size(X))*0.25;

nT = 10;
epsilon = 0.0001;
sigmaV = 0.5;
sigmaY = 0.1;
nIter = 1000;
X1 = lddmmLandmark(X, Y, sigmaV, sigmaY, nT,epsilon, nIter);