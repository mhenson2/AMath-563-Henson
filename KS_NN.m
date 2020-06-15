%Train a NN  for the KS equation
load('KS_train1.mat','uu','x','tt')
uu1 = uu;
x1 = x;
% load('KS_train2.mat','uu','x','tt')
% uu2 = uu;
% x2 = x;
input = uu1(:,1:end-1);
output = uu1(:,2:end);
% input = [uu1(:,1:end-1); uu2(:,1:end-1)];
% output = [uu1(:,2:end); uu2(:,2:end)];
%%
net = feedforwardnet([10 10 10]);
net.layers{1}.transferFcn = 'logsig';
net.layers{2}.transferFcn = 'radbas';
net.layers{3}.transferFcn = 'purelin';
net = train(net,input,output);
%save('KS_FF1')

%load('KS_FF1.mat')



clear ynn
x0 = input(:,1);
ynn = x0;

time = 0:0.4:150;
for jj=2:length(tt)
    y0=net(x0);
    ynn(:,jj)=y0; x0=y0;
end

figure(1)
surf(tt,x1,ynn), shading interp, colormap(hot), axis tight 
set(gca,'zlim',[-5 50])
xlabel('time')
ylabel('space')
zlabel('Neural Net Solution')
title('Neural Net')
%% 
figure(2)
surf(tt,x1,uu1), shading interp, colormap(hot), axis tight
set(gca,'zlim',[-5 50])
xlabel('time')
ylabel('space')
zlabel('KS Simulated Solution')
title('KS True Solution')
%% trying a different training technique
layers = [ ...
    imageInputLayer([28 28 1])
    convolution2dLayer(12,25)
    reluLayer
    fullyConnectedLayer(1)
    regressionLayer]
options = trainingOptions('sgdm', ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropFactor',0.2, ...
    'LearnRateDropPeriod',5, ...
    'MaxEpochs',20, ...
    'MiniBatchSize',64, ...
    'Plots','training-progress');
net = trainNetwork(input,output,layers,options);
save('KS_tN')
x = 32*pi*rand(N,1)/N;
x0 = cos(x/16).*(1+sin(x/16));
ynn(1,:)=x0';

time = 0:0.4:150;
for jj=2:length(tt)
    y0=net(x0);
    ynn(jj,:)=y0.'; x0=y0;
end
%% alternate method
load('KS_train1.mat','x','tt','uu')

data = uu;

numTimeStepsTrain = floor(0.9*numel(tt));

dataTrain = data(:,1:numTimeStepsTrain+1);
dataTest = data(:,numTimeStepsTrain:end);

%standardize data
for i = 1:128
    mu(i) = mean(dataTrain(i,:));
    sig(i) = std(dataTrain(i,:));
end

dataTrainStandardized = (dataTrain - mu'.*ones(128,226)); %./ sig;

for i = 1:128
    dataTrainStandardized(i,:) = dataTrainStandardized(i,:) / sig(i);
end

%% 

XTrain = dataTrainStandardized(:,1:end-1);
YTrain = dataTrainStandardized(:,2:end);

numFeatures = 128;
numResponses = 128;
numHiddenUnits = 200;

layers = [ ...
    sequenceInputLayer(numFeatures)
    lstmLayer(numHiddenUnits)
    fullyConnectedLayer(numResponses)
    regressionLayer];


options = trainingOptions('adam', ...
    'MaxEpochs',250, ...
    'GradientThreshold',1, ...
    'InitialLearnRate',0.005, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropPeriod',125, ...
    'LearnRateDropFactor',0.2, ...
    'Verbose',0, ...
    'Plots','training-progress');

net = trainNetwork(XTrain,YTrain,layers,options);
save('KS_net')
%% 

load ('KS_net')
dataTestStandardized = (dataTest - mu'.*ones(128,27)); %./ sig;

for i = 1:128
    dataTestStandardized(i,:) = dataTestStandardized(i,:) / sig(i);
end

XTest = dataTestStandardized(:,1:end-1);

net = predictAndUpdateState(net,XTrain);
[net,YPred] = predictAndUpdateState(net,YTrain(:,end));


numTimeStepsTest = length(tt) - numTimeStepsTrain ;
for i = 2:numTimeStepsTest
    [net,YPred(:,i)] = predictAndUpdateState(net,YPred(:,i-1),'ExecutionEnvironment','cpu');
end
%% unstandarize
YTest = dataTest(:,2:end);
for i = 1:128
    YPred(i,:) = sig(i)*YPred(i,:) + mu(i)*ones(1,26);
   rmse(i) = sqrt(mean(YPred(i,:) - YTest(i,:)).^2);
end

%% compare forcast

figure
%subplot(2,1,1)
surf(tt(numTimeStepsTrain+1:end),x,YTest), shading interp, colormap(hot), axis tight 
set(gca,'zlim',[-5 50]) 
hold on
surf(tt(numTimeStepsTrain+1:end),x,YPred)
hold off
legend(["True Solution" "Neural Net"])
ylabel("Cases")
title("Neural Net of KS Equation Against True Solution")
%% 
load('KS_test.mat')
N = 128;
XTrue = uu;
x = 16*pi*(1:N)'/N;
x0 = cos(x/16).*(1+sin(x/16));

x0standardized = (x0 -mean(x0)) / std(x0);

% net = predictAndUpdateState(net,XTrain);
% [net,YPred] = predictAndUpdateState(net,YTrain(:,end));
clear Ynn
Ynn(:,1) = x0standardized;

%numTimeStepsTest = length(tt) - numTimeStepsTrain ;
for i = 2:26
    Ynn(:,i) = predict(net,YPred(:,i-1));%,'ExecutionEnvironment','cpu');
end
% clear ynn
% ynn(:,1)=x0;
% for jj=2:3
%     y0=net(x0);
%     ynn(jj,:)=y0.'; x0=y0;
% end
%% unstandardized
Ynn = (Ynn - mean(x0))/std(x0);
%% 
figure
surf(tt(numTimeStepsTrain+1:end),x,uu(:,numTimeStepsTrain+1:end)), shading interp, colormap(hot), axis tight 
set(gca,'zlim',[-5 50]) 
hold on
surf(tt(numTimeStepsTrain+1:end),x,Ynn)
hold off
legend(["True Solution" "Neural Net"])
ylabel("Cases")
title("KS Prediction with Different Initial Conditions")
%% 
rmse = sqrt(mean((uu(:,numTimeStepsTrain+1:end)'-Ynn').^2));