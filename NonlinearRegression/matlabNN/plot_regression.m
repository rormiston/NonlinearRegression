function [ ] = plot_regression(bg_in, darm_in, noise_guess, fs)

    figure()
    time = (1:5000)*1.0/fs;
    leftover = darm_in - bg_in - noise_guess;
    plot(time, darm_in(1:5000)-bg_in(1:5000), 'k.',...
         time, noise_guess(1:5000), 'b',... 
         time, leftover(1:5000), 'r')
    legend('Orig. Noise', 'Noise guess', 'Leftover')
end

