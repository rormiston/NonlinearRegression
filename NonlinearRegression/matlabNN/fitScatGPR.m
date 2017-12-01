%% Load and Prepare Data
doo = 1;
if doo == 1

    load ScatData.mat

    x = x_input;
    t = x_darm;
else
    % load data made via the python MockData repo
    load ../../MockData/DARMwithNoise.mat

    tt = times;
    k = find(tt < 0.3, 1, 'last');

    ts = tt(2) - tt(1);
    fs = 1/ts;
    tt = tt(k:end);
    x  = wit(k:end);
    t       = 1e18 * darm(k:end);             % normalize DARM
    x_noise = 1e18 * background(k:end);

    clear darm wit1 wit2 background times
end

T = 1;
nfft = T*fs;
window = nuttallwin(nfft);

fmin = 10; fmax = 500;

MI = x'; SO = t';
ALL = [MI SO];
[nsamples,nch] = size(ALL);

[SOSpectraHold, ff] = calc_psd(SO,fs,window);
[numf,numt] = size(SOSpectraHold.fftamp);

psds = zeros(length(ff),nch);
ffts = zeros(length(ff),numt,nch);

for c = 1:nch
   [ALLSpectraHold1, ff] = calc_psd(ALL(:,c),fs,window);
   psds(:,c) = sqrt(ALLSpectraHold1.log_average);
   ffts(:,:,c) = ALLSpectraHold1.fftamp;
end

idx = find(ff >= fmin & ff <= fmax);
psds = psds(idx,:);
ffts = ffts(idx,:,:);

ff = ff(idx);

gprMdlsMag = {};
gprMdlsPhase = {};

originalSpectraHold = [];
predictedSpectraHold = [];
residualSpectraHold = [];

parpool(8);

%parfor i = 1:length(ff)
for i = 1:length(ff)
   if mod(i,10) == 0
      fprintf('%d/%d\n',i,length(ff));
   end

   MI = squeeze(ffts(i,:,2:end));
   SO = squeeze(ffts(i,:,1));

   MI_M = abs(MI); SO_M = abs(SO);    %magnitude
   MI_Ph = angle(MI); SO_Ph = angle(SO);  %phase angle
   X = [log10(MI_M) MI_Ph];

   %gprMdlsMag{i} = fitrgp(X,log10(SO_M),'KernelFunction','squaredexponential','FitMethod','exact','PredictMethod','exact','Standardize',1);

   gprMdlsMag{i} = fitrgp(X,log10(SO_M),'KernelFunction','squaredexponential','FitMethod','exact','PredictMethod','exact','Standardize',1,'OptimizeHyperparameters','auto','HyperparameterOptimizationOptions',struct('AcquisitionFunctionName','expected-improvement-plus'));
   predMag = predict(gprMdlsMag{i},X);
   predMag = 10.^predMag;

   %gprMdlsPhase{i} = fitrgp(X,SO_Ph,'KernelFunction','squaredexponential','FitMethod','exact','PredictMethod','exact','Standardize',1);

   gprMdlsPhase{i} = fitrgp(X,SO_Ph,'KernelFunction','squaredexponential','FitMethod','exact','PredictMethod','exact','Standardize',1,'OptimizeHyperparameters','auto','HyperparameterOptimizationOptions',struct('AcquisitionFunctionName','expected-improvement-plus'));
   predPhase = predict(gprMdlsPhase{i},X);

   predZ = predMag.*exp(predPhase*sqrt(-1));
   res = SO - predZ.';

   originalSpectraHold(i) = mean(abs(SO.^2));
   predictedSpectraHold(i) = mean(abs(predZ.^2));
   residualSpectraHold(i) = mean(abs(res.^2));

end

plotLocation = 'plots';

% Original and residual PSD plot
figure;
set(gcf, 'PaperSize',[10 6]);
set(gcf, 'PaperPosition', [0 0 10 6]);

semilogy(ff,originalSpectraHold,'k');
hold on
semilogy(ff,residualSpectraHold,'b');
semilogy(ff,predictedSpectraHold,'r');
hold off

xlabel('Frequency [Hz]');
ylabel('Power Spectral Density [unit/rtHz]');
legend_labels = {'Original','Residual','Predicted'};
hleg1 = legend(legend_labels);
set(hleg1,'Location','SouthWest')
set(hleg1,'Interpreter','none')

xlim([fmin fmax])

print('-dpng',[plotLocation '/psd.png']);
print('-depsc2',[plotLocation '/psd.eps']);
close;

