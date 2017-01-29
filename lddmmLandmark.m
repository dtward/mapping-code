% deform landmarks X to match Y
% cost function is \sum_i |phi X - Y|^2 (l2)
%
% derivation of gradient
% define a state q
% dynamics
% qdot_i = sum_j K(q_i,q_j) p_j
% enforce dynamics with lagrange muliplier
% gives 
% C = sum_ij K(q_0^i,q_0^j)p_0^iTp_0^j +  sum_i |q_1^i - y^i|^2 + int sum_i l^iT(qdot^i - sum_j K(q^i,q^j)p^j) dt
% extremize over q by taking variation dq with dq_0 = 0
% sum_i 2(q_1^i - y^i)dq^i_1 + int sum_i l^iT(dqdot^i - sum_j D_1K(q^i,q^j)dq^ip^j - D_2 K(q^i,q^j)dq^jp^j ) dt   
% sum_i 2(q_1^i - y^i)dq^i_1 - int sum_i ldot^iT dq^i dt - int sum_i sum_j l^iT D_1K(q^i,q^j)dq^ip^j + l^jT  D_2 K(q^j,q^i)dq^ip^i ) dt + sum_i l^iT_1 dq_1 
% sum_i 2(q_1^i - y^i)dq^i_1 - int sum_i ldot^iT dq^i dt - int sum_i sum_j l^iT D_1K(q^i,q^j)dq^ip^j + l^jT  D_1 K(q^i,q^j)dq^ip^i ) dt + sum_i l^iT_1 dq_1 
% sum_i 2(q_1^i - y^i)dq^i_1 - int sum_i ldot^iT dq^i dt - int sum_i sum_j (l^iTp^j+l^jT p^i)  D_1K(q^i,q^j)dq^i dt + sum_i l^iT_1 dq_1 
% for this to be zero for all dq we need
% l_1^i = -2(q_1^i - y^i)
% ldot^i = sum_j D_1 K^T(q^i,q^j)(l^jT p^i + l^iT p^j)

% now take perturbation with respect to p
% 2 sum_ij K(q_0^i,q_0^j)p_0^i dp_0^j - int sum_i l^iT sum_j K(q^i,q^j)dp^j dt
% for this to be zero for all dp we need
% 2 

function [X1] = lddmmLandmark(X,Y)

nT = 50;
N = size(X,1);
Pt = zeros(N,size(X,2),nT+1);
sigmaV = 1;
sigmaY = 0.01;
epsilon = 0.00005;


% start iterating
for iter = 1 : 1000
% flow forward
Xt = flowXForward(X,Pt,sigmaV);
E = calculateCost(Xt,Y,Pt,sigmaV,sigmaY);
disp(['Error : ' num2str(E)])


% gradient
Lambda1 = -(Xt(:,:,end) - Y)/sigmaY^2;
% flow backward
Lambdat = flowLambdaBackward(Xt,Pt,Lambda1,sigmaV);

% gradient descent
Pt = Pt - epsilon*(Pt - Lambdat);

Pt(:,:,1:end-1) - Lambdat(:,:,1:end-1)
disp(['Error : ' num2str(E)])



hold off;
for i = 1 : nT+1
scatter(Xt(:,1,i),Xt(:,2,i))
hold on;
end
scatter(Y(:,1),Y(:,2))
axis image
hold off;
drawnow;

end
X1 = Xt(:,:,end);

function Xt = flowXForward(X,Pt,sigmaV)
nT = size(Pt,3)-1;
dt = 1.0/nT;
Xt = zeros(size(Pt));
Xt(:,:,1) = X;
for i = 1 : nT
    K = exp(-pdist2(Xt(:,:,i),Xt(:,:,i)).^2/2.0/sigmaV^2);
    Xt(:,:,i+1) = Xt(:,:,i) + K*Pt(:,:,i)*dt;
end

function Lambdat = flowLambdaBackward(Xt,Pt,Lambda1,sigmaV)
nT = size(Pt,3)-1;
dt = 1.0/nT;
Lambdat = zeros(size(Pt));
Lambdat(:,:,end) = Lambda1;
for i = nT : -1 : 1
    K = exp(-pdist2(Xt(:,:,i),Xt(:,:,i)).^2/2.0/sigmaV^2);
    deltaX = bsxfun(@minus,Xt(:,1,i), Xt(:,1,i)');
    deltaY = bsxfun(@minus,Xt(:,2,i), Xt(:,2,i)');
    if size(Xt,2) == 3
        deltaZ = bsxfun(@minus,Xt(:,3,i), Xt(:,3,i)');  
    end
    lipj = Lambdat(:,:,i)*Pt(:,:,i)';
    ljpi = lipj';
    pipj = Pt(:,:,i)*Pt(:,:,i)';
    scalar = (-lipj-ljpi+pipj).*K*(-1/sigmaV^2);
    Lambdat(:,1,i) = Lambdat(:,1,i+1) - sum( scalar.*deltaX,2)*dt;
    Lambdat(:,2,i) = Lambdat(:,2,i+1) - sum( scalar.*deltaY,2)*dt;
    if size(Xt,2) == 3
        Lambdat(:,3,i) = Lambdat(:,3,i+1) - sum( scalar.*deltaZ,2)*dt;
    end
end

function E = calculateCost(Xt,Y,Pt,sigmaV,sigmaY)
nT = size(Pt,3)-1;
E = 0;
for i = 1 : nT
    K = exp(-pdist2(Xt(:,:,i),Xt(:,:,i)).^2/2/sigmaV^2);
    qdot = K*Pt(:,:,i);
    E = E + sum(sum(qdot.*Pt(:,:,i)))/2/nT;
end
E = E + sum(sum((Xt(:,:,end) - Y).^2))/2/sigmaY^2;