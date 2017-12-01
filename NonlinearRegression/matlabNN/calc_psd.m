function [psd, ff] = homestake_psd(samples,samplef,window)
%
%

nfft = length(window);

ff = linspace(0,samplef/2,nfft/2+1);
ff(end) = [];

oT = length(samples)/samplef; %observation time
sT = floor(nfft/samplef);
specnum = floor(oT/sT+0.0000001); %number of spectra calculated from data

normspec = norm(window)^2/nfft;

psd.part = zeros(nfft/2,specnum);
psd.fftamp = zeros(nfft/2,specnum);

% ------ calculate FFT amplitudes ------------------------------      
for k=1:specnum
   try
      timestretch = detrend(samples((1+(k-1)*nfft):(k*nfft)),'constant');
      fftamp = fft(timestretch.*window);
      chanamp = fft(timestretch.*window) /samplef;
      psd.part(:,k) = 2/sT*abs(chanamp(1:floor(nfft/2))).^2/normspec;
      psd.fftamp(:,k) = fftamp(1:floor(nfft/2));
   catch
   end
end

try
   psd.mean_val = mean(samples); psd.average = mean(psd.part,2);
   psd.log_average = 10.^mean(log10(psd.part(:,any(psd.part))),2); 
catch
   psd = [];
end


