function [cleandata,x]=denoiseKSVDversion2(noisydata,noise,param)
disp('Starting to  train the  dictionary');


cleandata = noisydata - noise;


[Dictionary,output]  = KSVD(cleandata,param);
theta = output.CoefMatrix;
Psi = Dictionary ;
x =Psi*theta;

% figure(1)
% imagesc(noisydata);
% figure(2)
% imagesc(cleandata);
end