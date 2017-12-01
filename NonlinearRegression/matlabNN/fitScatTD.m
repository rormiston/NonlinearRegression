%% Load and Prepare Data
doo = 1;
if doo == 0

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
    x  = wit(:, k:end);
    t       = 1e18 * darm(k:end);             % normalize DARM
    x_noise = 1e18 * background(k:end);

    clear darm wit1 wit2 background times
end

% filter target and background data so that the network can focus 
% on the frequencies that can be improved instead of loud low freq noise
[b,a] = butter(2, [40 400]/(fs/2));
% Save unfiltered versions for plotting
t_full = t;
bg_full = x_noise;
t = filtfilt(b, a, t);
x_noise = filtfilt(b, a, x_noise);

% convert matrices to sequences for timeseries-based network 
x_seq = con2seq(x);
t_seq = con2seq(t);

%% Set up time delay network 

% how many points of "lag" to use as inputs to the network
n_delay = 25;

% make network and set stopping conditions
slow_factor = 4;
slow_delays = [1:slow_factor:n_delay*slow_factor];

timedelay_net = timedelaynet(slow_delays, 5);
% timedelay_net.divideFcn = '';
timedelay_net.trainParam.min_grad = 1e-6;
timedelay_net.trainParam.goal = 0.005;
timedelay_net.trainParam.epochs = 300;

timedelay_net.divideParam.trainRatio = 50/100;
timedelay_net.divideParam.valRatio   = 10/100;
timedelay_net.divideParam.testRatio  = 40/100;
% generate sequence of input vectors by sliding window 
[x_shifted,x_i,Ai,t_shifted] = preparets(timedelay_net,x_seq, t_seq);

% Choose Plot Functions
% For a list of all plot functions type: help nnplot
timedelay_net.plotFcns = {'plotperform','ploterrhist', ...
                          'plotregression', 'plotfit'};

% slow_factor = 128;
% slow_delays = [1:slow_factor:n_delay*slow_factor];
% timedelay_net_dummy = timedelaynet(slow_delays, 5);
% [x_slow,x_i_slow,Ai_slow,t_slow] = preparets(timedelay_net_dummy,x_seq, t_seq);
% 
% x_i_mat = seq2con(x_i_slow);
% x_i_mat = x_i_mat{1,1};
% x_mat = seq2con(x_slow);
% x_mat = x_mat{1,1};
% x_mat_all = cat(2, x_i_mat, x_mat);

%% Run network and plot results
% train network: should make pop-up window
timedelay_net = train(timedelay_net,x_shifted,t_shifted,x_i, Ai);


% get timeseries of nonlinear noise based on final network conditions
% slightly shorter than input timeseries because it needs n_delay points 
% of history before its first output
noise_guess = cell2mat(sim(timedelay_net,x_shifted,x_i));

% plot results, adjusting other inputs to be the same length as noise_guess
diff = size(bg_full)-size(noise_guess);
diff = diff(2)
size(noise_guess)
size(bg_full(1+diff:end))
size(bg_full)
plot_sub_results(bg_full(1+diff:end), t_full(1+diff:end), noise_guess, fs)
plot_regression(x_noise(1+diff:end), t(1+diff:end), noise_guess, fs)
