% deform landmarks X to match Y
% cost function is \sum_i |phi X - Y|^2 (l2)
% we use hamiltonian method to derive gradient

function [X1,Pt] = lddmmInitialMomentumLandmark(X,Y,sigmaV,sigmaY,nT,epsilon,nIter)

% nT = 10;
N = size(X,1);
P0 = zeros(size(X));
Pt = zeros(N,size(X,2),nT+1);
% sigmaV = 1;
% sigmaY = 0.01;
% epsilon = 0.00005;
% nIter = 1000;


% start iterating
for iter = 1 : nIter
% flow forward
[Xt,Pt] = flowXForward(X,P0,sigmaV,nT);
E = calculateCost(Xt,Y,P0,sigmaV,sigmaY);
fprintf('Iter %d of %d, energy is %g\n',iter,nIter,E);

% Endpoint gradient
LambdaQ1 = -(Xt(:,:,end) - Y)/sigmaY^2;
LambdaP1 = zeros(size(LambdaQ1));

% flow backward
Lambdat = flowLambdaBackward(Xt,Pt,LambdaQ1,LambdaP1,sigmaV,nT);

% gradient descent
Pt = Pt - epsilon*(Pt - Lambdat);

% Pt(:,:,1:end-1) - Lambdat(:,:,1:end-1)
% disp(['Error : ' num2str(E)])
% hold off;
% for i = 1 : nT+1
% scatter(Xt(:,1,i),Xt(:,2,i))
% hold on;
% end
% scatter(Y(:,1),Y(:,2))
% axis image
% hold off;
% drawnow;

end
X1 = Xt(:,:,end);

function [Xt,Pt] = flowXForward(X0,P0,sigmaV,nT)
dt = 1.0/nT;
Xt = zeros(size(X0,1),size(X0,2),nT+1);
Xt(:,:,1) = X0;
Pt = zeros(size(X0,1),size(X0,2),nT+1);
Pt(:,:,1) = P0;
PDot = zeros(size(X0));

for i = 1 : nT
    K = exp(-pdist2(Xt(:,:,i),Xt(:,:,i)).^2/2.0/sigmaV^2);
    deltaX = bsxfun(@minus,Xt(:,1,i), Xt(:,1,i)');
    deltaY = bsxfun(@minus,Xt(:,2,i), Xt(:,2,i)');
    if size(X0,2) == 3
        deltaZ = bsxfun(@minus,Xt(:,3,i), Xt(:,3,i)');  
    end
    
    pipj = Pt(:,:,i)*Pt(:,:,i)';    
    scalar = -pipj.*K/sigmaV;
    Pdot(:,1) = sum(scalar.*deltaX,2);
    Pdot(:,2) = sum(scalar.*deltaY,2);
    if size(X0,2) == 3
        Pdot(:,3) = sum(scalar.*deltaZ,2);
    end
    
    Xt(:,:,i+1) = Xt(:,:,i) + K*Pt(:,:,i)*dt;
    Pt(:,:,i+1) = Pt(:,:,i) + Pdot*dt;
end

function [LambdaQt,LambdaPt] = flowLambdaBackward(Xt,Pt,LambdaQ1,LambdaP1,sigmaV,nT)
dt = 1.0/nT;
LambdaQt = zeros(size(Pt));
LambdaQt(:,:,end) = LambdaQ1;
LambdaPt = zeros(size(Pt));
LambdaPt(:,:,end) = LambdaP1;

lambdaQdot = zeros(size(LambdaQ1));
lambdaPdot = zeros(size(LambdaQ1));

for i = nT : -1 : 1
    K = exp(-pdist2(Xt(:,:,i),Xt(:,:,i)).^2/2.0/sigmaV^2);
    deltaX = bsxfun(@minus,Xt(:,1,i), Xt(:,1,i)');
    deltaY = bsxfun(@minus,Xt(:,2,i), Xt(:,2,i)');    
    if size(Xt,2) == 3
        deltaZ = bsxfun(@minus,Xt(:,3,i), Xt(:,3,i)');  
    end
    
    deltaLambdaPX = bsxfun(@minus,lambdaPt(:,1,i),lambdaPt(:,1,i)');
    deltaLambdaPY = bsxfun(@minus,lambdaPt(:,2,i),lambdaPt(:,2,i)');
    if size(Xt,2) == 3
        deltaLambdaPY = bsxfun(@minus,lambdaPt(:,3,i),lambdaPt(:,3,i)');
    end
    
    lQipj = LambdaQt(:,:,i)*Pt(:,:,i)';
    lQjpi = lQipj';
    pipj = Pt(:,:,i)*Pt(:,:,i)';
    
    lQp = lQipj + lQjpi;
    
    % put together the first term of lqdot
    scalar = lQp.*K*(-1/sigmaV^2);
    lambdaQDot(:,:,1) = sum(scalar.*deltaX,2);
    lambdaQDot(:,:,2) = sum(scalar.*deltaY,2);
    if size(Xt,2)==3
        lambdaQDot(:,:,3) = sum(scalar.*deltaZ,2);
    end
    
    % the second term
    KL = deltaX.*deltaLambdaPX + deltaY.*deltaLambdaPY;
    if size(Xt,2)==3
        KL = KL + deltaZ.*deltaLambdaPZ;
    end
    scalar = pipj.*K.*(1/sigmaV^4).*KL; % second derivatives
    lambdaQDot(:,:,1) = lambdaQDot(:,:,1) + sum(scalar.*deltaX,2);
    lambdaQDot(:,:,2) = lambdaQDot(:,:,2) + sum(scalar.*deltaY,2);
    if size(Xt,2)==3
        lambdaQDot(:,:,3) = lambdaQDot(:,:,3) + sum(scalar.*deltaZ,2);
    end
    
    % now we work on lambdapdot
    
    keyboard
    
    
    
    
    Lambdat(:,1,i) = Lambdat(:,1,i+1) - sum( scalar.*deltaX,2)*dt;
    Lambdat(:,2,i) = Lambdat(:,2,i+1) - sum( scalar.*deltaY,2)*dt;
    if size(Xt,2) == 3
        Lambdat(:,3,i) = Lambdat(:,3,i+1) - sum( scalar.*deltaZ,2)*dt;
    end
end

function E = calculateCost(Xt,Y,P0,sigmaV,sigmaY)
E = 0;
K = exp(-pdist2(Xt(:,:,1),Xt(:,:,1)).^2/2/sigmaV^2);
qdot = K*P0;
E = E + sum(sum(qdot.*P0))/2;
E = E + sum(sum((Xt(:,:,end) - Y).^2))/2/sigmaY^2;