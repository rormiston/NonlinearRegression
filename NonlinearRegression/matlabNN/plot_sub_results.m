function [ ] = plot_sub_results(bg, darm, noise_guess, fs)
    nfft = 2*fs;
    [Pdarm,ff]  = pwelch(darm, hann(nfft), nfft/2, nfft, fs);
    [Pest, ff]  = pwelch(noise_guess, hann(nfft), nfft/2, nfft, fs);
    [Pshot,ff]  = pwelch(bg, hann(nfft), nfft/2, nfft, fs);
    [Pres, ff]  = pwelch(darm-noise_guess, hann(nfft), nfft/2, nfft, fs);
    leftover = (darm-noise_guess) - bg;
    [Pleft, ff]  = pwelch(leftover, hann(nfft), nfft/2, nfft, fs);


    figure()
    loglog(ff, sqrt(Pdarm), 'k',...
           ff, sqrt(Pest), 'c',...
           ff, sqrt(Pres), 'r',...
           ff, sqrt(Pleft), 'b', ... 
           ff, sqrt(Pshot), 'g--', ...
           'LineWidth', 3)
    grid
    xlabel('Frequency [Hz]')
    ylabel('cts /rHz')
    legend('DARM', 'Estimate', 'Residual', 'Leftover', 'Shot Noise',...
           'Location', 'SouthWest')
    axis([10 500 1e-4 1e1])
    set(findall(gcf,'type','axes'),'fontsize',14)
    set(findall(gcf,'type','text'),'fontSize',14) 

     
end


