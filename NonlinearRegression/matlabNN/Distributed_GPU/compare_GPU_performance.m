
% Code to compare Distributed GPU Performance boost to Neural Network Scattering codes
% 24th Jan 2017, Nikhil

% Run in Caltech LDAS PCDEV1 Cluster: gsissh ldas-pcdev1.ligo.caltech.edu
% Run using Matlab_R2016b:  /ldcg/matlab_r2016b/bin/matlab

%% Parameter Set
pSet = {'no','yes'};% Parallel options
gSet = {'no','yes'};% GPU options
dSet = {16,32,64,256,512,1024,2048};  % Data length in Seconds (Fs=1000Hz, check mkScatNoise2.m)
%dSet = {5,10,30};
wSet = {12}; % MATLAB Parallel Worker (Max=12)
iter = 10;   % Repeat calculations for better Mean Estimate
dCutoff = 512; % Cutoff to stop the original non-parallelized non-GPU code at some data length to save time (in sec, say 256)
legendSet = {'Parallel Off, GPU Off','Parallel Off, GPU On','Parallel On, GPU Off','Parallel On, GPU On'};% Figure Legend

% Close Current figure
close all
figure(1); clf

% Open Parallel Pool of Workers
if 1
delete(gcp('nocreate'));
pool = parpool('local');
end

% Make GPU Result Folder
if ~exist('GPU_Results') == 7
mkdir('GPU_Results')
end


% Start Comparison Runs
for pToggleIDX = 1:length(pSet)

  for gToggleIDX = 1:length(gSet)
  

dCount = 1;
clear lenData timeData


    for dToggleIDX = 1:length(dSet)
   
     for wToggleIDX = 1:length(wSet)

count = 1;
clear timeIDX tIDX yIDX eIDX
pToggle = pSet{pToggleIDX};
gToggle = gSet{gToggleIDX};
dToggle = dSet{dToggleIDX};
wToggle = wSet{wToggleIDX};

if  (  ( ~strcmp(pToggle,'no') + ~strcmp(gToggle,'no')) + ( strcmp(pToggle,'no').*strcmp(gToggle,'no').*(dToggle <=dCutoff) )  ) > 0 

       for idx = 1:iter
      
disp(sprintf(['Parallel_%s_GPU_%s_DataSamples_%d_PWorkers_%d_iter_%d'],pToggle,gToggle,dToggle,wToggle,idx))

    % Generate Data Set
    mkScatNoise2(dToggle); 
    load ScatData.mat;
    x = x_input;
    t = x_darm;


tic;

%% Choose a Training Function
% For a list of all training functions type: help nntrain
% 'trainlm' is usually fastest.
% 'trainbr' takes longer but may be better for challenging problems.
% 'trainscg' uses less memory. Suitable in low memory situations.
trainFcn = 'trainscg';  % trainlm : Bayesian Regularization backpropagation.

% Create a Fitting Network
hiddenLayerSize = [7];                       % this is just a guess
net = fitnet(hiddenLayerSize, trainFcn);

% Choose Input and Output Pre/Post-Processing Functions
% For a list of all processing functions type: help nnprocess
net.input.processFcns  = {'removeconstantrows','mapminmax'};
net.output.processFcns = {'removeconstantrows','mapminmax'};

% Setup Division of Data for Training, Validation, Testing
% For a list of all data division functions type: help nndivision
net.divideFcn  = 'divideblock';  % Divide data randomly
net.divideMode = 'sample';       % Divide up every sample
net.divideParam.trainRatio = 50/100;
net.divideParam.valRatio   = 10/100;
net.divideParam.testRatio  = 40/100;

                                               

% Choose a Performance Function
% For a list of all performance functions type: help nnperformance
net.performFcn = 'mse';  % Mean Squared Error

% Choose Plot Functions
% For a list of all plot functions type: help nnplot
net.plotFcns = {'plotperform','plottrainstate','ploterrhist', ...
                'plotregression', 'plotfit'};

[b,a] = butter(2, [40 400]/(fs/2));
t_full = t;
bg_full = x_noise;
t = filtfilt(b, a, t);
x_noise = filtfilt(b, a, x_noise);


% Many MATLAB functions automatically execute on a GPU when any of the input arguments is a gpuArray. Normally you move arrays to and from the GPU with the functions gpuArray and gather. However, for neural network calculations on a GPU to be efficient, matrices need to be transposed and the columns padded so that the first element in each column aligns properly in the GPU memory. Neural Network Toolbox provides a special function called nndata2gpu to move an array to a GPU and properly organize it:
xg = nndata2gpu(x);
tg = nndata2gpu(t);
net2 = configure(net,x,t);



% On GPUs and other hardware where you might want to deploy your neural networks, it is often the case that the exponential function exp is not implemented with hardware, but with a software library. This can slow down neural networks that use the tansig sigmoid transfer function. An alternative function is the Elliot sigmoid function whose expression does not include a call to any higher order functions:
for i=1:net2.numLayers
  if strcmp(net2.layers{i}.transferFcn,'tansig')
    net2.layers{i}.transferFcn = 'elliotsig';
  end
end


% Training & Prediction
[net2,tr] = train(net2,x,t,'useParallel',pToggle,'useGPU',gToggle,'showResources','yes');
yg = net2(xg,'useParallel',pToggle,'useGPU',gToggle,'showResources','yes');
y = gpu2nndata(yg);



% Estimate Error
e = gsubtract(t,y);
performance = perform(net,t,y);

% Recalculate Training, Validation and Test Performance
trainTargets = t .* tr.trainMask{1};
valTargets   = t .* tr.valMask{1};
testTargets  = t .* tr.testMask{1};
trainPerformance = perform(net,trainTargets,y);
valPerformance   = perform(net,valTargets,y);
testPerformance  = perform(net,testTargets,y);

timeIDX(idx) = toc;
testPerfIDX(idx) = testPerformance;
tIDX(idx,:) = t;
yIDX(idx,:)  = y;
eIDX(idx,:)  = e;


end % iter loop ends



% Get Mean Execution Time
tfinal = mean(timeIDX);


% Select Best obtained Performance
[~,mID] = min(testPerfIDX);
targetFinal = tIDX(mID,:);
predFinal   = yIDX(mID,:);
errorFinal = eIDX(mID,:);

% Save Results 
DATA.Type = sprintf(['Parallel_%s_GPU_%s_DataSamples_%d_PWorkers_%d'],pToggle,gToggle,dToggle,wToggle);
DATA.tfinal = tfinal;
DATA.targetFinal = targetFinal;
DATA.predFinal = predFinal;
DATA.errorFinal = errorFinal;
%DATA.target  = t;
%DATA.prediction = y;
%DATA.error = e;
%DATA.trainPerformance = trainPerformance;
%DATA.valPerformance  = valPerformance ;
%DATA.testPerformance = testPerformance;
fileName = sprintf(['./GPU_Results/Parallel_%s_GPU_%s_DataSamples_%d_PWorkers_%d.mat'],pToggle,gToggle,dToggle,wToggle);
save(fileName,'DATA');


lenData(dCount)  = dToggle;
timeData(dCount) = tfinal;

end % cutoff for non-parallel, non-GPU case 

end % wToggle loop ends

dCount  = dCount+1;

end % dToggle loop ends

% Generate Figure ( Done within the loop so that results can be inspected during  code execution)
fig1 = figure(1);
plot(lenData,timeData,'o-','linewidth',2);
xlabel('Data duration [Sec] with Fs=1000Hz ')
ylabel('Execution time [Sec]')
title('fitScat.m Performance')
grid on
hold on
legend(gca,legendSet,'Location','NorthWest')
saveas(fig1,'GPU_Performance.pdf')
saveas(gca,'GPU_Performance.fig')

end % gToggle loop ends
end % pToggle loop ends

legend(gca,legendSet,'Location','NorthWest');
saveas(fig1,'GPU_Performance.pdf')
saveas(gca,'GPU_Performance.fig')

copyfile('GPU_Performance.pdf','./GPU_Results/')
copyfile('GPU_Performance.fig','./GPU_Results/')
