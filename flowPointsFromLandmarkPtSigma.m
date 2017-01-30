function Yt = flowPointsFromLandmarkPtSigma(X0,Pt,sigmaV,Y)
% flow the points Y
% given the initial points X0, and time varying momentum PT

nT = size(Pt,3)-1;
dt = 1.0/nT;
Xt = zeros(size(Pt));
Xt(:,:,1) = X0;


Yt = zeros([size(Y,1),size(Y,2),nT+1]);
Yt(:,:,1) = Y;

for i = 1 : nT
    % kernel for updating Y
    KY = exp(-pdist2(Yt(:,:,i),Xt(:,:,i)).^2/2.0/sigmaV^2);
    Yt(:,:,i+1) = Yt(:,:,i) + KY*Pt(:,:,i)*dt;
    
    K = exp(-pdist2(Xt(:,:,i),Xt(:,:,i)).^2/2.0/sigmaV^2);
    Xt(:,:,i+1) = Xt(:,:,i) + K*Pt(:,:,i)*dt;
end
