function [x,Psi,v]=denoiseKSVDNNversion3(noisydata,param)
disp('Starting to  train the  dictionary');

[Dictionary,output]  = KSVD_NN(noisydata,param);
thetawithnoise = output.CoefMatrix;
Psi = Dictionary ;
[row,col,v]=find(thetawithnoise);
num = size(v);
noise= mean(v)*rand(num);
% lambda = 2;
% noise = mean(v) *poissrnd(lambda,num)
v_new = v - noise;
for i = 1:1:num
    theta(row(i),col(i))=v_new(i);
end
x =Psi*theta;
end