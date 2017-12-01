%% Load and Prepare Data
doo = 0;
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

% filter target and background data so that the network can focus 
% on the frequencies that can be improved instead of loud low freq noise
[b,a] = butter(2, [40 400]/(fs/2));
% Save unfiltered versions for plotting
t_full = t;
bg_full = x_noise;
t = filtfilt(b, a, t);
x_noise = filtfilt(b, a, x_noise);

x_seq = con2seq(x);
t_seq = con2seq(t);

%% Set up NARX network 

% how many points of "lag" to use as inputs to the network
delays = [1:5];
narx_net = narxnet(delays, delays, 5);
narx_net.divideFcn = '';
narx_net.trainParam.min_grad = 1e-6;
narx_net.trainParam.goal = 0.005;
narx_net.trainParam.epochs = 50;
% generate sequence of input vectors by sliding window 
[x_shifted,x_i,Ai,t_shifted] = preparets(narx_net,x_seq,{},t_seq);

%% Run network and plot results
% train network: should make pop-up window. This version uses the target
% as input to the network alongside the witnesses. 
narx_net = train(narx_net,x_shifted,t_shifted,x_i);

% get timeseries of nonlinear noise based on final network conditions
% slightly shorter than input timeseries because it needs n_delay points 
% of history before its first output
noise_guess = cell2mat(sim(narx_net,x_shifted,x_i));

% plot results, adjusting other inputs to be the same length as noise_guess
plot_sub_results(bg_full(1+n_delay:end), t_full(1+n_delay:end), noise_guess, fs)

