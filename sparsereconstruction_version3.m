close all;
clc;
clear;

date='20160729_24.14';
material='oil';
sizeOfMaterial='big';


distance = 0.20;
% distance = 0.214777;
% path=fullfile('D:','WuJie','experiment data','compressed sensing',date);
% path= fullfile('D:','wangsuyuan','24GHz signal processing','experiment data','compressed sensing',date);
path= fullfile('\\tsclient\D\wangsuyuan\24GHz signal processing\experiment data\compressed sensing\',date);

allFile=dir(fullfile(path,[date,'.',material,'.',sizeOfMaterial,'.value.mat']));
dataFile=allFile.name;
load(fullfile(path,dataFile));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
cw1 = num2str(f1/1e9);
cw2 = num2str(f2/1e9);
lambda1 = 3e8/f1;
lambda2 = 3e8/f2;

% deltaPhi1 =distance*4*pi*(f2-f1)/3e8; % when phi1 > phi2
% degreePhi1=180*deltaPhi1/pi;
% deltaPhi2 =(distance-3e8/(2*(f2-f1)))*4*pi*(f2-f1)/3e8+2*pi;% when phi1< phi2
% %deltaPhi2 =(distance-3e8/(2*(f2-f1)))*4*pi*(f2-f1)/3e8;% when phi1< phi2
% degreePhi2=180*deltaPhi2/pi;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%             I_value_channel = I_value(:,channel)  ;
%             Q_value_channel = Q_value(:,channel) ;
n=size(I_value_ch1,1);
numberOfFile=n;
N=size(I_value_ch1,2);
K=150;%atoms number of dictionary
M=n;%the rescovery of X
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
param.K = K; % number of dictionary elements
param.numIteration =10; % number of iteration to execute the K-SVD algorithm.
% param.errorFlag =1; % decompose signals until a certain error is reached. do not use fix number of coefficients.
% param.errorGoal = 0;
param.errorFlag =0; % decompose signals until a certain error is reached. do not use fix number of coefficients.
param.L = 6;
param.preserveDCAtom = 0;
%%%%%%%% initial dictionary: Dictionary elements %%%%%%%%
param.InitializationMethod =  'DataElements';
% param.initialDictionary = randn(n,param.K );
% % param.initialDictionary = I_value_ch1(:,1:param.K );% 
% param.InitializationMethod =  'GivenMatrix';
% param.displayProgress = 1;
% SNR_I=1;  SNR_Q=1;
% SNRdB_I=10*log(SNR_I);  SNRdB_Q=10*log(SNR_Q); 
% 
% I=double(I)+128;
% I=uint8(I); 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% OriImage=imread('lena.jpg');
% sigma = 1.6;
% grayImg=rgb2gray(OriImage);
% gausFilter = fspecial('gaussian',[5 5],sigma);
% blur=imfilter(grayImg,gausFilter,'replicate');
%  imshow(blur)

% sigma=floor(mean(std(I_value_ch1)));
% hsize=3*sigma;
% gausFilter = fspecial('gaussian',hsize,sigma);
% blur=imfilter(I_value_ch1,gausFilter,'replicate');
% figure(1)
% imagesc(I_value_ch1);
% figure(2)
% imagesc(blur)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% I_value_ch1_prime = im2uint8(double(I_value_ch1));
% imwrite(I_value_ch1,'I_value_ch1.jpg');
% imwrite(I_value_ch1_prime,'I_value_ch1_prime.jpg');
% 
% image = imread('I_value_ch1.jpg');
% image_prime = imread('I_value_ch1_prime.jpg');
% image_double = im2double(image);
% figure(1)
% imshow(image_double);
% figure(2)
% imagesc(I_value_ch1);
% figure(3)
% imagesc(image_double);
%  I_value_ch1_green = I_value_ch1_color(:,:,2);
% result = (image==image_prime);
% 
% I_value_ch1_blue
%  
% imagesc(I_value_ch1);
% imwrite(I_value_ch1,'I_value_ch2.jpg');
% imwrite(I_value_ch1,'Q_value_ch1.jpg');
% imwrite(I_value_ch1,'Q_value_ch2.jpg');
% 
% figure(2);
% subplot(2,2,1);imshow(I_value_ch1,[255]); title('I_1');
% subplot(2,2,2);imshow(Q_value_ch1,[255]); title('Q_1');
% subplot(2,2,3);imshow(I_value_ch2,[255]); title('I_2');
% subplot(2,2,4);imshow(Q_value_ch2,[255]); title('Q_2');
I_value_ch1_mean =mean(I_value_ch1,2);%avearage of columns
Q_value_ch1_mean =mean(Q_value_ch1,2);%avearage of columns
I_value_ch2_mean =mean(I_value_ch2,2);%avearage of columns
Q_value_ch2_mean =mean(Q_value_ch2,2);%avearage of columns

I_value_ch1_sigma=floor(mean(std(I_value_ch1)));
I_value_ch2_sigma=floor(mean(std(I_value_ch2)));
Q_value_ch1_sigma=floor(mean(std(Q_value_ch1)));
Q_value_ch2_sigma=floor(mean(std(Q_value_ch2)));
% 
% I_value_ch1_noise = random('norm', I_value_ch1_mean, I_value_ch1_sigma, [numberOfFile 1]);
% I_value_ch1_noise = repmat(I_value_ch1_noise,1,N);
% I_value_ch2_noise = random('norm', I_value_ch2_mean, I_value_ch2_sigma, [numberOfFile 1]);
% I_value_ch2_noise = repmat(I_value_ch2_noise,1,N);
% Q_value_ch1_noise = random('norm', Q_value_ch1_mean, Q_value_ch1_sigma, [numberOfFile 1]);
% Q_value_ch1_noise = repmat(Q_value_ch1_noise,1,N);
% Q_value_ch2_noise = random('norm', Q_value_ch2_mean, Q_value_ch2_sigma, [numberOfFile 1]);
% Q_value_ch2_noise = repmat(Q_value_ch2_noise,1,N);

% [cleandata,x]=denoiseKSVDversion2(noisydata,noise,param)
% [x]=denoiseKSVDversion3(noisydata,param)
% [x,Psi,v]=denoiseKSVDNNversion3(noisydata,param)
[I_value_ch1_reconstructed,I_value_ch1_Psi,I_value_ch1_theta]=denoiseKSVDversion3(I_value_ch1,param);
[I_value_ch2_reconstructed,I_value_ch2_Psi,I_value_ch2_theta]=denoiseKSVDversion3(I_value_ch2,param);
[Q_value_ch1_reconstructed,Q_value_ch1_Psi,Q_value_ch1_theta]=denoiseKSVDNNversion3(Q_value_ch1,param);
[Q_value_ch2_reconstructed,Q_value_ch2_Psi,Q_value_ch2_theta]=denoiseKSVDNNversion3(Q_value_ch2,param);
% 
% [Q_value_ch1_Phi,Q_value_ch1_A,Q_value_ch1_reconstructed]=denoiseKSVDNN(Q_value_ch1,SNRdB_Q,param);
% 
% [I_value_ch2_Phi,I_value_ch2_A,I_value_ch2_reconstructed]=denoiseKSVDNN(I_value_ch2,SNRdB_I,param);
% 
% [Q_value_ch2_Phi,Q_value_ch2_A,Q_value_ch2_reconstructed]=denoiseKSVDNN(Q_value_ch2,SNRdB_Q,param);

figure(1);
subplot(2,2,1);imagesc(I_value_ch1); title('I_1');
subplot(2,2,2);imagesc(Q_value_ch1); title('Q_1');
subplot(2,2,3);imagesc(I_value_ch2); title('I_2');
subplot(2,2,4);imagesc(Q_value_ch2); title('Q_2');



figure(2);
subplot(2,2,1);imagesc(I_value_ch1_reconstructed); title('I_1reconstructed');
subplot(2,2,2);imagesc(Q_value_ch1_reconstructed); title('Q_1reconstructed');
subplot(2,2,3);imagesc(I_value_ch2_reconstructed); title('I_2reconstructed');
subplot(2,2,4);imagesc(Q_value_ch2_reconstructed); title('Q_2reconstructed');

I_value_ch1_reconstructed_mean =mean(I_value_ch1_reconstructed,2);%avearage of columns
Q_value_ch1_reconstructed_mean =mean(Q_value_ch1_reconstructed,2);%avearage of columns
I_value_ch2_reconstructed_mean =mean(I_value_ch2_reconstructed,2);%avearage of columns
Q_value_ch2_reconstructed_mean =mean(Q_value_ch2_reconstructed,2);%avearage of columns

I_value_ch1_mean =mean(I_value_ch1,2);%avearage of columns
Q_value_ch1_mean =mean(Q_value_ch1,2);%avearage of columns
I_value_ch2_mean =mean(I_value_ch2,2);%avearage of columns
Q_value_ch2_mean =mean(Q_value_ch2,2);%avearage of columns

% I_ave_channel(:,1) = I_value_ch1_mean;
% I_ave_channel(:,2) = I_value_ch2_mean;
% Q_ave_channel(:,1) = Q_value_ch1_mean;
% Q_ave_channel(:,2) = Q_value_ch2_mean;

% a(1,channel) = 0;
% b(1,channel) = 0;
% r(1,channel)=mean(x);
I_ave_channel=[I_value_ch1_mean  I_value_ch2_mean];
Q_ave_channel=[Q_value_ch1_mean  Q_value_ch2_mean];
I_real_channel=[I_value_ch1_reconstructed_mean  I_value_ch2_reconstructed_mean];
Q_real_channel=[Q_value_ch1_reconstructed_mean  Q_value_ch2_reconstructed_mean];


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    s=['rbgykmc']; %颜色属性  channel1 is red, channel2 is blue
    u=['*odx+vp']; % 点标记属性
for channel=1:2

    figure(3)
    for i = 1:numberOfFile
        plot(I_ave_channel(i,channel),Q_ave_channel(i,channel),[s(channel),u(channel)]);
        title(['Original','       ',  material, '\_', sizeOfMaterial,'\_',num2str(cw1),'GHz\_',num2str(cw2),'GHz']);
        xlabel('I');
        ylabel('Q');
        hold on
        phase_original(i,channel)=atan2(Q_ave_channel(i,channel),I_ave_channel(i,channel));
    end
    % theta = (-pi:0.05*pi:pi)';
    % plot(r(1,channel) *cos(theta)+a(1,channel) ,r(1,channel) *sin(theta)+b(1,channel) ,[s(channel)]);
    xlim([-100 100]);
    ylim([-100 100]);
    axis equal;
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%            
%             I_real_channel=r(1,channel) *cos(atan2(Q_ave_channel - b(1,channel) ,I_ave_channel-a(1,channel) ))+a(1,channel) ;
%             Q_real_channel= r(1,channel) *sin(atan2(Q_ave_channel - b(1,channel) ,I_ave_channel-a(1,channel) ))+b(1,channel) ;
%             I_real(:,channel)=I_real_channel;
%             Q_real(:,channel)=Q_real_channel;

    figure(4)
    for i = 1:numberOfFile
            plot(I_real_channel(i,channel),Q_real_channel(i,channel),[s(channel),u(channel)]);
            title(['Removing noise','       ',  material, '\_', sizeOfMaterial,'\_',num2str(cw1),'GHz\_',num2str(cw2),'GHz']);

%                     title(['Removing noise','       ',date, '\_',  material, '\_', sizeOfMaterial]);
            xlabel('I');
            ylabel('Q');
            hold on
            phase_adjust(i,channel)=atan2(Q_real_channel(i,channel),I_real_channel(i,channel));
    end
%             plot(r(1,channel)*cos(theta)+a(1,channel),r(1,channel)*sin(theta)+b(1,channel),[s(channel)]);hold on;
        plot(0,0,'ko'); hold on;
        xlim([-100 100]);
        ylim([-100 100]);
        axis equal;

end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
phase_value_ch1 = phase_original(:,1);
phase_value_ch2 = phase_original(:,2);
phase_value_ch1_reconstructed=phase_adjust(:,1);
phase_value_ch2_reconstructed=phase_adjust(:,2);
I_channel1_original = I_ave_channel(:,1);
I_channel2_original = I_ave_channel(:,2);
Q_channel1_original =Q_ave_channel(:,1);
Q_channel2_original =Q_ave_channel(:,2);
I_channel1=I_real_channel(:,1);
I_channel2=I_real_channel(:,2);
Q_channel1=Q_real_channel(:,1);
Q_channel2=Q_real_channel(:,2);

phase_value_ch1 = phase_original(:,1);
phase_value_ch2 = phase_original(:,2);
phase_value_ch1_reconstructed=phase_adjust(:,1);
phase_value_ch2_reconstructed=phase_adjust(:,2);

deltaPhi = phase_value_ch1 - phase_value_ch2;
if deltaPhi < 0
    F=2*pi*lambda1/(lambda1-lambda2)+(lambda1.*phase_value_ch1 -lambda2.*phase_value_ch2 )/(lambda1-lambda2);
else
    F=(lambda1.*phase_value_ch1 -lambda2.*phase_value_ch2 )/(lambda1-lambda2);
end

deltaPhi_reconstructed =phase_value_ch1_reconstructed - phase_value_ch2_reconstructed;
if  deltaPhi_reconstructed <0
    F_reconstructed=2*pi*lambda1/(lambda1-lambda2)+(lambda1.*phase_value_ch1_reconstructed -lambda2.*phase_value_ch2_reconstructed )/(lambda1-lambda2);
else
    F_reconstructed=(lambda1.*phase_value_ch1_reconstructed -lambda2.*phase_value_ch2_reconstructed )/(lambda1-lambda2);
end


% filename =fullfile(path,[date,'.',material,'.',sizeOfMaterial,'.value.csv']);
% % filename = 'D:\wangsuyuan\result\20160802.empty.csv';
% head = {'I1';'I2';'Q1';'Q2';'I1fitted';'I2fitted';'Q1fitted';'Q2fitted';'phase1';'phase2';'F'};
% values =[I_channel1_original;I_channel2_original;Q_channel1_original;Q_channel2_original;...
%     I_channel1;I_channel1;Q_channel1;Q_channel2;phase_channel1;phase_channel2;F];
% data = table(I_channel1_original,I_channel2_original,Q_channel1_original,Q_channel2_original,...
%     I_channel1,I_channel1,Q_channel1,Q_channel2,phase_channel1,phase_channel2,F,...
%     'VariableNames', head);
% writetable(data,filename);