clc;
clear all;
close all;

addpath('G:\dsp\Linear_time_invariant_system\exampleDTMF');

numberSamples = 2000;
samplingFreq = 50000; % We assume here a sampling frequency of 50KHz

timeStamps = (1:numberSamples) / samplingFreq;

ampMsg = 1; % Amplitude of the modulating signal (sinusoid)
ampCar = 1; % Amplitude of the carrier signal (sinusoid)
freqMsg = 440; % Frequency of the modulating sinusoid: a simple A note
freqCar = 5000; % Frequency of the carrier: 5KHz, with freqCar>freqMsg
modIndex = 0.5; % Modulation index

signalMsg = ampMsg * sin(2 * pi * freqMsg * timeStamps); % Message Signal

signalCar = ampCar * sin(2 * pi * freqCar * timeStamps); % Carrier Signal

signalModulated = (1 + modIndex * signalMsg) .* signalCar; % Amplitude Modulated signal

windowSamples = 1:500; % Select a visualization window composed of the first 500 samples

figure;

% modulating signal
subplot(3,1,1);
plot(signalMsg(windowSamples));
% some easthetical settings
set(gca, 'fontsize', 14);
ylabel('signalMsg');
title('Message Signal ', 'FontSize', 20);
set(gca, 'fontsize', 14);
hLabel = get(gca, 'YLabel');
set(hLabel, 'FontSize', 20);

% Carrier signal
subplot(3,1,2);
plot(signalCar(windowSamples));
% Some easthetical settings
set(gca, 'fontsize', 14);
ylabel('signalCar');
title('Carrier Signal ', 'FontSize', 20);
hLabel = get(gca, 'YLabel');
set(hLabel, 'FontSize', 20);

%Modulated signal
subplot(3,1,3);
plot(signalModulated(windowSamples));
% some easthetical settings
set(gca, 'fontsize', 14);
xlabel('samples ');
ylabel('signalModulated ');
title('AM Signal', 'FontSize', 20); 
hLabel = get(gca, 'YLabel');
set(hLabel, 'FontSize', 20);

hLabel = get(gca, 'XLabel');
set(hLabel, 'FontSize', 20);

SignalMsg = fft(signalMsg); % Compute the DFT/DFS of the modulating signal
SignalCar = fft(signalCar); % compute the DFT/DFS of the carrier signal
SignalModulated = fft(signalModulated); % Compute the DFT/DFS of the modulated signal

normFreq = (1:numberSamples) / numberSamples; % normalized frequencies

figure;

% Spectrum of the modulating signal
subplot(3,1,1);
plot(normFreq, abs(SignalMsg));
% some easthetical settings
set(gca, 'fontsize', 14);
ylabel('|SignalMsg| ');
title('Magnitude Message Spectrum ', 'FontSize', 20);
set(gca, 'fontsize', 14);
hLabel = get(gca, 'YLabel');
set(hLabel, 'FontSize', 20);

% Spectrum of the carrier signal
subplot(3,1,2);
plot(normFreq, abs(SignalCar));
% Some easthetical settings
set(gca, 'fontsize', 14);
ylabel('|SignalCar| ');
title('Magnitude Carrier Spectrum ', 'FontSize', 20);
hLabel = get(gca, 'YLabel');
set(hLabel, 'FontSize', 20);

% Spectrum of the modulated signal
subplot(3,1,3);
plot(normFreq, abs(SignalModulated));
% Some easthetical settings
set(gca, 'fontsize', 14);
xlabel('normalized frequencies ');
ylabel('|SignalModulated| ');
title('Magnitude AM Spectrum ', 'FontSize', 20);
hLabel = get(gca, 'YLabel');
set(hLabel, 'FontSize', 20);

hLabel = get(gca, 'XLabel');
set(hLabel, 'FontSize', 20);


%% Mixer Signal
ampMix = 1;
freqMix = 5000;

signalMix = ampMix * sin(2 * pi * freqMix * timeStamps);

%% Mixed the generated mixer signal and the modulated signal together
signalIF = signalMix .* signalModulated;

%{
% Construct the impulse response of the system of length 100
M = 10;
lambda = (M - 1) / M;
h = (1-lambda)*lambda.^(0:99); 

y = conv2(signalIF, h, 'valid');
%}


%% Filter signal
LAMBDA = 0.98;

omega0 = 2*pi*freqMsg/samplingFreq;
b = 1;
a = [1; -2*LAMBDA*cos(omega0); LAMBDA^2];
y = my_filter(signalIF, b, a);


%y = (y - 1) / modIndex;
y = y ./ 150;

figure;

% IF signal
subplot(3,1,2);
plot(y(windowSamples));
% Some easthetical settings
set(gca, 'fontsize', 14);
%ylabel('IF signal');
title('IF Signal ', 'FontSize', 20);
hLabel = get(gca, 'YLabel');
set(hLabel, 'FontSize', 20);


SignalIF = fft(y);

figure;

% Spectrum of the IF signal
subplot(3,1,1);
plot(normFreq, abs(SignalIF));
% some easthetical settings
set(gca, 'fontsize', 14);
ylabel('|SignalMsg| ');
title('Magnitude IF Spectrum ', 'FontSize', 20);
set(gca, 'fontsize', 14);
hLabel = get(gca, 'YLabel');
set(hLabel, 'FontSize', 20);


ICoherentSum = zeros(size(timeStamps));
for i = 1:4
  sigma2 = 3; %power of the noise
  noise = sigma2*randn(size(timeStamps)); % Gaussian noise
  y = y .+ noise';

  %% Coherent Signal
  ampCoh = 1;
  phi = pi / 2;
  signalCoh = ampCoh * sin(2 * pi * freqMsg * timeStamps + phi);

  mulSignal = y' .* signalCoh;
  ICoherentSum = ICoherentSum + mulSignal;

end

figure;

% Coherent Sum signal
subplot(3,1,2);
plot(ICoherentSum(windowSamples));
% Some easthetical settings
set(gca, 'fontsize', 14);
%ylabel('Coherent Sum signal');
title('Coherent Sum Signal ', 'FontSize', 20);
hLabel = get(gca, 'YLabel');
set(hLabel, 'FontSize', 20);




