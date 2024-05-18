%Matlab code relate to (1) "Reinforcement learning and approximate Bayesian computation for model selection and parameter calibration applied to a nonlinear dynamical system, Ritto, T.G., Beregi, S.,Barton, D.A.W., Mechanical Systems and Signal Processing, 2022, 181, 109485."

%Code applied to a simple spring problem:

clear all
close all
clc
rand('seed',200) 
randn('seed',200) 

%% Parameters
nMC=250;% number of samples
M=2; % number of models
epsilon=0.1; % error accepted in ABC
cv=0.3; % coef of variation of k
% cv=1;
cv1=0.3; % coef of variation of k1
cv2=0.3; % coef of variation of k2

%%
% Reference (ideally experimental)
kref=1e6;
%kref=0.5;
x=linspace(1,3,20);
% noise
sigma=5e5;
noise=normrnd(0,sigma,length(x),1)';
% reference response
fref=kref*x.^2+noise;

figure1 = figure;
axes1 = axes('Parent',figure1);
plot(x,fref,'.')

%%
% model #1
k=1.2e6;
f1=k*x.^2;
hold on
plot(x,f1,'-k')

%%
% model #2
k1=1.1e6;
k2=0.3e6;
f2=k1*x+k2*x.^3;
hold on
plot(x,f2,'--k')
xlabel('x','fontsize',14)
ylabel('f','fontsize',14)
legend('ref','prior model 1','prior model 2')
set(axes1,'FontSize',14);
grid on


%%
% initiating parameters of the beta_ distribution
% Thompson Sampling (reinforcement learning) to select the model
alpha_=[2;2];
beta_=[2;2];

% initiating with zero reward
reward=zeros(nMC,M);

% initiating erromin (big value)
erromin1=1e6;
erromin2=1e6;

cont1=1;
cont2=1;
krand=0;
k1rand=0;
k2rand=0;
kid=k;
k1id=k1;
k2id=k2;

l_krefdin = linspace(0.5,2,nMC);

%%
for i=1:nMC
    
    %noise=normrnd(0,sigma,length(x),1)';
    %krefdin=l_krefdin(i);
    %fref=krefdin*x.^2+noise;

    % ABC
    %for 1=1:20

    %end
    

    % sample from beta_
    theta1=betarnd(alpha_(1),beta_(1));
    theta2=betarnd(alpha_(2),beta_(2));
    theta=[theta1;theta2];
    
    % select model with bigger theta
    [M,I] = max(theta);
    % pull this arm and collect reward
    if I==1
        %ka=normrnd(k,k*cv);
        ka=unifrnd(k-k*cv,k+k*cv);
        f1a=ka*x.^2;
        if norm(f1a-fref)^2/norm(fref)^2<epsilon
            krand(cont1)=ka;
            cont1=cont1+1;
            if norm(f1a-fref)^2/norm(fref)^2<erromin1
                erromin1=norm(f1a-fref)^2/norm(fref)^2;
                kid=ka;
            end
            r=1;
            reward(i,I)=r;
            alpha_(I)=alpha_(I)+r;
            beta_(I)=beta_(I)+(1-r);
        else
            r=0;
            alpha_(I)=alpha_(I)+r;
            beta_(I)=beta_(I)+(1-r);
        end
        
    elseif I==2
        %k1a=normrnd(k1,k1*cv1);
        %k2a=normrnd(k2,k2*cv2);
        k1a=unifrnd(k1-k1*cv1,k1+k1*cv1);
        k2a=unifrnd(k2-k2*cv2,k2+k2*cv2);
        f2a=k1a*x+k2a*x.^3;
        if norm(f2a-fref)^2/norm(fref)^2<epsilon
           k1rand(cont2)=k1a;
           k2rand(cont2)=k2a;
           cont2=cont2+1;
           if norm(f2a-fref)^2/norm(fref)^2<erromin2
               erromin2=norm(f2a-fref)^2/norm(fref)^2;
               k1id=k1a;
               k2id=k2a;
           end
           r=1;
           reward(i,I)=r;
           alpha_(I)=alpha_(I)+r;
           beta_(I)=beta_(I)+(1-r);
        else
            r=0;
            alpha_(I)=alpha_(I)+r;
            beta_(I)=beta_(I)+(1-r);
        end
    end

    P1(i)=sum(reward(:,1))/i;
    P2(i)=sum(reward(:,2))/i;

end

Hor=linspace(0,nMC,nMC);
acceptance_M1=1-(nMC-length(krand))/nMC
acceptance_M2=1-(nMC-length(k1rand))/nMC

% approximating the posteriors
Npto=1000;
[pk,xx]=ksdensity(krand,linspace(k-k*cv,k+k*cv,Npto),'npoints',Npto);
[pk1,xx1]=ksdensity(k1rand,linspace(k1-k1*cv1,k1+k1*cv1,Npto),'npoints',Npto);
[pk2,xx2]=ksdensity(k2rand,linspace(k2-k2*cv2,k2+k2*cv2,Npto),'npoints',Npto);

% beta distributions
bxx=linspace(0,1,Npto);
beta_orig = 1/beta(2,2)*(bxx).^(2-1).*(1-bxx).^(2-1);
beta_m1 = 1/beta(alpha_(1),beta_(1))*(bxx).^(alpha_(1)-1).*(1-bxx).^(beta_(1)-1);
beta_m2 = 1/beta(alpha_(2),beta_(2))*(bxx).^(alpha_(2)-1).*(1-bxx).^(beta_(2)-1);

%%

%figure
figure1 = figure;
axes1 = axes('Parent',figure1);
plot(Hor,P1,'linewidth',1.5)
hold on
plot(Hor,P2,'linewidth',1.5)
xlabel('n')
ylabel('reward')
legend('model #1','model #2')
set(axes1,'FontSize',14);
grid on


%figure
figure1 = figure;
axes1 = axes('Parent',figure1);
plot(x,fref,'.')

% model #1
f1=kid*x.^2;
hold on
plot(x,f1,'-k')

% model #2
f2=k1id*x+k2id*x.^3;
hold on
plot(x,f2,'--k')
xlabel('x','fontsize',14)
ylabel('f','fontsize',14)
legend('ref','model 1','model 2')
set(axes1,'FontSize',14);
grid on

figure1 = figure;
axes1 = axes('Parent',figure1);
%histogram(normrnd(k,k*cv,nMC,1),'Normalization','pdf')
%histogram(unifrnd(k-k*cv,k+k*cv,nMC,1),'Normalization','pdf')
%histogram(krand,'FaceColor','g','Normalization','pdf')
plot(xx,pk,'-k','linewidth',1.5)
hold on
plot([k-k*cv k+k*cv],[1/(k+k*cv-(k-k*cv)) 1/(k+k*cv-(k-k*cv))],'--k','linewidth',1.5)
hold on
plot([k-k*cv k-k*cv],[0 1/(k+k*cv-(k-k*cv))],'--k','linewidth',1.5)
hold on
plot([k+k*cv k+k*cv],[0 1/(k+k*cv-(k-k*cv))],'--k','linewidth',1.5)
xlabel('k','fontsize',14)
title('Model 1: k x^2','fontsize',14)
legend('posterior','prior')
set(axes1,'FontSize',14);
grid on

figure1 = figure;
axes1 = axes('Parent',figure1);
plot(xx1,pk1,'-k','linewidth',1.5)
hold on
plot([k1-k1*cv1 k1+k1*cv1],[1/(k1+k1*cv1-(k1-k1*cv1)) 1/(k1+k1*cv1-(k1-k1*cv1))],'--k','linewidth',1.5)
hold on
plot([k1-k1*cv1 k1-k1*cv1],[0 1/(k1+k1*cv1-(k1-k1*cv1))],'--k','linewidth',1.5)
hold on
plot([k1+k1*cv1 k1+k1*cv1],[0 1/(k1+k1*cv1-(k1-k1*cv1))],'--k','linewidth',1.5)
xlabel('k_1','fontsize',14)
title('Model 2: k_1 x + k_2 x^3','fontsize',14)
legend('posterior','prior')
set(axes1,'FontSize',14);
grid on

figure1 = figure;
axes1 = axes('Parent',figure1);
plot(xx2,pk2,'-k','linewidth',1.5)
hold on
plot([k2-k2*cv2 k2+k2*cv2],[1/(k2+k2*cv2-(k2-k2*cv2)) 1/(k2+k2*cv2-(k2-k2*cv2))],'--k','linewidth',1.5)
hold on
plot([k2-k2*cv2 k2-k2*cv2],[0 1/(k2+k2*cv2-(k2-k2*cv2))],'--k','linewidth',1.5)
hold on
plot([k2+k2*cv2 k2+k2*cv2],[0 1/(k2+k2*cv2-(k2-k2*cv2))],'--k','linewidth',1.5)
xlabel('k_2','fontsize',14)
title('Model 2: k_1 x + k_2 x^3','fontsize',14)
legend('posterior','prior')
set(axes1,'FontSize',14);
grid on

figure1 = figure;
axes1 = axes('Parent',figure1);
plot(bxx,beta_orig,'--k','linewidth',1.5)
hold on
plot(bxx,beta_m1,'linewidth',1.5)
hold on
plot(bxx,beta_m2,'linewidth',1.5)
xlabel('support','fontsize',14)
legend('prior','model #1','model #2')
grid on
