% make some test data for testing various Feedforward networks
function  mkScatNoise2(dToggle)

% dToggle = duration in seconds 

lambda = 1064e-9;  % laser wavelength
fs = 1000; % Hz
ts = 1/fs; % s
dur = dToggle;  % s
tt = 0:ts:dur;
tt = tt(1:end-1);

% make some seismic noise
x = randn(dur * fs,1);
[b,a] = cheby1(4, 4, 4/(fs/2));  % low pass filter for seismic noise
x_seis = 0.5e-6 * filter(b,a,x);
x_seis = x_seis';

% the chamber door is shaking at 60 Hz by 0.1 nm
x_wall = 0.8e-7 * sin(2*pi*60*tt);
rand_phase = rand;
x_wall_coup = 0.8e-7 * sin(2*pi*(60*tt + rand_phase));

x_noise = 0.2 * randn(dur * fs,1)';

x_darm = sin(2*(2*pi/lambda)*(x_seis + x_wall_coup)) + x_noise;

x_input = [x_seis; x_wall];
save ScatData x_input x_darm x_noise tt fs ts

%% plot some things
if 1
    figure(922413)
    nfft = 2*fs;
    pwelch(x_darm, hann(nfft), nfft/2, nfft, fs)
end
