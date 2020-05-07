%Develop a DMD model
%import data
D = importdata('pelt_data.csv');

x1 = D.data(:,2);
x2 = D.data(:,3);

X = [x1' ; x2'];
%% 
%interpolate data
t = 1:2:60;
dt = 0.5;
xq = 1:dt:60;
s = spline(t,x1,xq);
s2 = spline(t,x2,xq);
% figure(1)
% plot(t,x1,'o')
% hold on 
% plot(xq,s)

Y = [s;s2];
%% 
% run DMD
r = 2;

[Phi,Lambda] = DMD(Y(:,1:end-1),Y(:,2:end),r);

t = 1:dt:100;
mu = diag(Lambda);
omega = log(mu)/dt;
u0 = Y(:,1);
y0 = Phi\u0; %psuedo inverse IC
u_modes = zeros(r,length(t));
for i = 1:length(t)
    u_modes(:,i) = (y0.*exp(omega*t(i)));
end
u_dmd1 = Phi*u_modes;
figure(2)
plot(t, u_dmd1(1,:),t,u_dmd1(2,:))%, xq, s,'o', xq,s2,'o')
%plot(t(592:end), u_dmd1(1,592:end),t(592:end),u_dmd1(2,592:end))
legend('Hare Population', 'Lynx Population')
title('DMD Model')
%hold on
%plot(t,s,t,s2)
%% 
%time delay build Heinkle matrix
H = [];
Hdot = [];
for j = 1:25
    H = [H; Y(:,j:(50+j))];
    Hdot = [Hdot; Y(:,(j+1):(50+j+1))];
end
[u,s1,v]= svd(H);
figure(3)
plot(diag(s1)/sum(diag(s1)), 'o')
%subplot(2,1,1), plot(u(:,1:3),'Linewidth',[2])
%subplot(2,1,2), plot(v(:,1:3),'Linewidth',[2])
%% 
r = 17;

[Phih,Lambdah] = DMD(H,Hdot,r);
t = 1:dt:100;
muh = diag(Lambdah);
omegah = log(muh)/dt;
u0h = H(:,1);
y0h = Phih\u0h; %psuedo inverse IC
u_modes2 = zeros(r,length(t));
for i = 1:length(t)
    u_modes2(:,i) = (y0h.*exp(omegah*t(i)));
end
u_dmd2 = Phih*u_modes2;
figure(4)
subplot(2,1,1)
plot(t,u_dmd2(1,:),xq,s)
title('Time Delay DMD Model')
legend('time delay DMD model for hare population', 'true interpolated hair population')
%hold on
%plot(t,x1,'o')

%hold off
%plot(diag(s)/sum(diag(s)), 'o')
subplot(2,1,2)
plot(t,u_dmd2(2,:),xq,s2)
legend('time delay DMD model for lynx population', 'true interpolated lynz population')

%hold on
%plot(t,x2,'o')
%hold off


%% fit parameters for lotka volterra

b = 0.5;
p = 0.1;
r = 0.1;
d = 2;

IC = X(:,1);
tspan = [0 60];
[t1,y] = ode45(@(t1,y) LotVol(t1,y,b,p,r,d),tspan,IC);

subplot(2,1,1), plot(t1,y(:,1)), hold on
plot(xq,s), hold off
title('Lotka-Volterra')
legend('model hare population','true hare population')
subplot(2,1,2), plot(t1,y(:,2)), hold on
plot(xq,s2), hold off
%title('Lotka-Volterra')
legend('model lynx population','true lynx population')
% subplot(3,1,3), plot(t1,y(:,1),t1,y(:,2))
% 
%% 


%model discovery

tz = 1:dt:60;
n = length(tz);

clear x2dot, clear x1dot

for k = 2:n-1
    x1dot(k-1) = (Y(1,k+1) - Y(1,k-1))/(2*dt);
    x2dot(k-1) = (Y(2,k+1) - Y(2,k-1))/(2*dt);
end

% for k = 2:n-1
%     x1dot(k-1) = (x1(k+1) - x1(k-1))/(2*ddt);
%     x2dot(k-1) = (x2(k+1) - x2(k-1))/(2*ddt);
% end
x1s = Y(1,2:n-1)';
x2s= Y(2,2:n-1)';

Th = [ones(length(x1s),1) x1s x2s x1s.^2 x2s.^2 x1s.*x2s x1s.^3 x2s.^3 (x1s.^2).*x2s x1s.*x2s.^2]; %sin(x1s) sin(x2s) cos(x1s) cos(x2s)];

xi1 = lasso(Th,x1dot,'Lambda',0.2,'CV',10);
xi2 =  lasso(Th,x2dot,'Lambda',0.2,'CV',10);

figure(6)
B = categorical({'1','x','y','x^2','y^2','xy','x^3','y^3','x^2y','xy^2'});%'sin(x)','sin(y)','cos(x)','cos(y)'});
B = reordercats(B,{'1','x','y','x^2','y^2','xy','x^3','y^3','x^2y','xy^2'});%,'sin(x)','sin(y)','cos(x)','cos(y)' });
subplot(2,1,1), bar(B, xi1)
title('Parameters for SINDy Model of Hare Population')

subplot(2,1,2), bar(B,xi2)
title('Parameters for SINDy Model of Lynx Population')

disMod1 = Th*xi1;
disMod2 = Th*xi2;
disMod = [disMod1'; disMod2'];

figure(7)
subplot(2,1,1)
plot(tz(2:n-1),disMod1), hold on, plot(xq,s), hold off
title('SINDy Model')
legend('model hare population','true hare population')
subplot(2,1,2)
plot(tz(2:n-1), disMod2), hold on, plot(xq,s2), hold off
%title('SINDy Model')
legend('model lynx population','true lynx population')


%% 
%p = length(u_dmd1);
%KL divergence
%range = 1:2:60;
f1 = hist(Y(1,:));%generate pdfs
f2 = hist(Y(2,:));
g1 = hist(real(u_dmd1(1,1:119)));%DMD
g2 = hist(real(u_dmd1(2,1:119)));
h1 = hist(real(u_dmd2(1,1:119)));%DMD time delay
h2 = hist(real(u_dmd2(2,1:119)));
k1 = hist(y(1,:));%LV
k2 = hist(y(2,:)); 
m1 = hist(disMod1);%SINDy
m2 = hist(disMod2);

%g3 = hist
%normalize data
f1 = f1/trapz(f1) + .01;
g1 = g1/trapz(g1) + .01;
g2 = g2/trapz(g2)+ .01;
h1 = h1/trapz(h1) + .01;
h2 = h2/trapz(h2)+ .01;
k1 = k1/trapz(k1) + .01;
k2 = k2/trapz(k2)+ .01;
m1 = m1/trapz(m1) + .01;
m2 = m2/trapz(m2)+ .01;

%compute integrand
int1 = f1.*log(f1./g1);
int2 = f2.*log(f2./g2);
int3 = f1.*log(f1./h1);
int4 = f2.*log(f2./h2);
int5 = f1.*log(f1./k1);
int6 = f2.*log(f2./k2);
int7 = f1.*log(f1./m1);
int8 = f2.*log(f2./m2);

%KL divergence
I1 = trapz(int1);
I2 = trapz(int2);
I3 = trapz(int3);
I4 = trapz(int4);
I5 = trapz(int5);
I6 = trapz(int6);
I7 = trapz(int7);
I8 = trapz(int8);
I = [I1;I2;I3;I4;I5;I6;I7;I8];
model = {'DMD Hare';'DMD Lynx';'Time-Delay-DMD Hare';'Time-Delay-DMD Lynx';'LV Hare'; 'LV Lynx';'SINDy Hare';'SINDy Lynx'};
KL = table(I,'RowNames',model,'VariableNames',{'KL-Divergence'})


%% AIC BIC
% mu1 = mean(u_dmd1(1,1:119));
% mu2 = mean(u_dmd2(1,1:119));
% mu3 =  mean(disMod1);
phat1 = mle(real(u_dmd1(1,1:119)));
phat2 = mle(real(u_dmd2(1,1:119)));
phat3 = mle(real(disMod1));
mu1 = phat1(1);
mu2 = phat2(1)
mu3 = phat3(1);

sig1 = phat1(2)^2;
sig2 = phat2(2)^2
sig3 = phat3(2)^2;

n = length(u_dmd1);
logL1 = -(n/2)*log(2*pi) - (n/2)*log(sig1) - (1./(2*sig1)).*real(sum((u_dmd1(1,1:119) - mu1).^2'))';
logL2 = -(n/2)*log(2*pi) - (n/2)*log(sig2) - (1./(2*sig2)).*real(sum((u_dmd2(1,1:119) - mu2)).^2')';
m = length(disMod1);
logL3 = -(m/2)*log(2*pi) - (m/2)*log(sig2) - (1./(2*sig3)).*real(sum((disMod1 - mu3)).^2')';
logL = [logL1;logL2;logL3];

K = [size(u_dmd1,1)*size(u_dmd1,2);size(u_dmd2,1)*size(u_dmd2,2);2];
aic = 2*K - 2*logL;
nn = [119;119;117];
bic = log(nn).*K - 2*logL;

model1 = {'DMD Hare';'Time Delay DMD Hare'; 'SINDy Hare'};
AB = table(aic, bic, 'RowNames',model1,'VariableNames', {'AIC';'BIC'})
