%load('reaction_diffusion_big.mat','t','x','y','u','v')
load('reacDiff.mat')
%%

u_mat = reshape(u,[],201);
v_mat = reshape(v,[],201);

load('reaction_diffusion_train1')
u1 = reshape(u,[],201);
v1 = reshape(v,[],201);
%% 

data = [u_mat; u1; v_mat; v1];

[U,S,V] = svd(data,'econ');
plot(diag(S)/sum(diag(S)),'o')
%% 
r = 12;
Ur = U(:,1:r);
data = Ur'*data;

%% neural net for u
input = data(:,1:end-1);
output = data(:,2:end);
net1 = feedforwardnet([10 10 10]);
net1.layers{1}.transferFcn = 'logsig';
net1.layers{2}.transferFcn = 'radbas';
net1.layers{3}.transferFcn = 'purelin';
net1 = train(net1,input',output');
save('RD_net')

%% 
load('reaction_diffusion_test.mat')

%testing dT
u_test = reshape(u,[],201);
v_test = reshape(v,[],201);
X0 = [u_test;v_test];


[U0,S0,V0] = svd(X0,'econ');
%% 
r = 12;
X0 = U0(:,1:r)'*X0;
%% intial conition SVD
%load('RD_net')

x0 = X0(1,1:end-1)';
clear RD_nn
RD_nn(1,:) = x0';
for jj = 2:12
    y0 = net1(x0);
    RD_nn(jj,:)=y0; x0=y0;
end

%% project back
RD_nn = U0(:,1:r)*RD_nn;

%% 
M = 512^2;
u_test = reshape(RD_nn(1:M,:),512,512,200);
v_test = reshape(RD_nn(M+1:end,:),512,512,200);
%% 
figure(2)
pcolor(x,y,u_test(:,:,end)); shading interp; colormap(hot)
title('Forecast from Low Rank Training u variable')

u = reshape(u_mat,512,512,201);
figure(3)
pcolor(x,y,u(:,:,end)); shading interp; colormap(hot)
title('True Solution of u')

%% figure(2)
figure(4)
pcolor(x,y,v_test(:,:,end)); shading interp; colormap(hot)
title('Forecast from Low Rank Training in v')

v = reshape(v_mat,512,512,201);
figure(5)
pcolor(x,y,v(:,:,end)); shading interp; colormap(hot)
title('True Solution of v')
