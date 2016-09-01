function [x,Psi,v]=denoiseKSVDversion3(noisydata,param)
disp('Starting to  train the  dictionary');

[Dictionary,output]  = KSVD(noisydata,param);
thetawithnoise = output.CoefMatrix;
Psi = Dictionary ;
[row,col,v]=find(thetawithnoise);
num = size(v);
noise= mean(v)*rand(num);
v_new = v - noise;
for i = 1:1:num
    theta(row(i),col(i))=v_new(i);
end
x =Psi*theta;

% figure(1)
% imagesc(noisydata);
% figure(2)
% imagesc(cleandata);
end