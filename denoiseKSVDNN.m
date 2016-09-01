function [Phi,Dictionary,cleandata]=denoiseKSVDNN(noisydata,SNRdB,param)
disp('Starting to  train the  dictionary');

noise = randn(size(noisydata));
actualNoise = calcNoiseFromSNR(SNRdB,noisydata, noise);
cleardata = noisydata - actualNoise;
Phi =eye(size(noisydata,1))+actualNoise*pinv(cleardata);

[Dictionary,output]  = KSVD_NN(noisydata,param);
theta = output.CoefMatrix;
Psi = inv(Phi)*Dictionary ;
cleandata =Psi*theta;

figure(1)
imagesc(noisydata);
figure(2)
imagesc(cleandata);
end