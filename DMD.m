function [Phi,Lambda] = DMD(X,Xprime,r)
[U, Sigma, V] = svd(X,'econ'); %step 1
Ur = U(:,1:r);
Sigmar = Sigma(1:r,1:r);
Vr = V(:,1:r);

%step 2
Atilde = Ur'*Xprime*Vr/Sigmar;

%step 3
[W, Lambda] = eig(Atilde);

%step 4
Phi = Xprime*Vr/Sigmar*W; %DMD modes
%alpha1 = Sigmar*(Vr(1,:)');
%b = (W*Lambda)\alpha1;


end