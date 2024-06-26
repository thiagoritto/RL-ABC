%Matlab code relate to (2) Reinforcement learning and approximate Bayesian computation (RL-ABC) for model selection and parameter calibration of time-varying systems, Ritto, T.G., Beregi, S., Barton, D.A.W., Mechanical Systems and Signal Processing, 2023, 200, 110458.

%This is just the structure of the strategy; it cannot be run!!!

clear all
close all
clc
format short

%% Define dataset

omega=24; % fixed frequency (Hz) used in the experiment (24 or 25)

changedataset=1; % if changedataset=1 datasets will change accordingly
dataset1=6;
dataset2=2;
dataset3=0; % if dataset3=0 run the first two datasets only
              
% starting with dataset1
dataset=dataset1;
% load experiment and run deterministic identification (opimisation algorithm - ex fmincon function)
run_dataset
run_identification



%% Parameters obtained in the deterministic identification
%model parametes [ mu, nu, rho, b, ugain, wn]
modparRef1 = modpar_id1
modparRef2 = modpar_id2
modparRef3 = modpar_id3

%defining bounds around the identified values
lbound=0.98;
ubound=1.02;
nMC=2000; % number of samples
M=3; % number of models
epsilon=0.005; % error accepted in ABC
stoprewarding=1; % if stoprewarding=0, do not used the stragegy of stop rewarding if beta distributions are too far appart
memory = nMC; % short memory effect: considering only the last 'memory' points to update Beta distribution
              % if memory = nMC: no memory effect is considered

% defining lower and upper bound values

modparRef1_ll=modparRef1*lbound;
modparRef1_uu=modparRef1*ubound;
modparRef2_ll=modparRef2*lbound;
modparRef2_uu=modparRef2*ubound;
modparRef3_ll=modparRef3*lbound;
modparRef3_uu=modparRef3*ubound;

% fixing gain (this is a parameter that will not be identified in RL-ABC)
modparRef1_ll(5)=modparRef1(5);
modparRef1_uu(5)=modparRef1(5);
modparRef2_ll(5)=modparRef2(5);
modparRef2_uu(5)=modparRef2(5);
modparRef3_ll(5)=modparRef3(5);
modparRef3_uu(5)=modparRef3(5);

% initiating parameters of the beta_ distribution
% Thompson Sampling (reinforcement learning) to select the model
alpha_=2*ones(1,M);
beta_=2*ones(1,M);

% used in the strategy of not letting the models die out
alpha_cont=2*ones(1,M);
beta_cont=2*ones(1,M);

%%
% initiating erromin (big value)
erromin=1e6*ones(1,M);

% initiating with zero reward
reward=zeros(nMC,M);

cont=zeros(M,1);
contACC=zeros(M,1);
globalACC=zeros(1,nMC);
Pts_analytic=length(yd);
xbr_mod1_id = zeros(1,Pts_analytic);
xbr_mod2_id = zeros(1,Pts_analytic);
xbr_mod3_id = zeros(1,Pts_analytic);
xbr_mod1_ale = zeros(1,Pts_analytic);
xbr_mod2_ale = zeros(1,Pts_analytic);
xbr_mod3_ale = zeros(1,Pts_analytic);
p_mod1=zeros(1,length(modparRef1));
p_mod2=zeros(1,length(modparRef2));
p_mod3=zeros(1,length(modparRef3));

contdataset1=0;
contdataset2=0;
contdataset3=0;
contr1=1;
contr2=1;
contr3=1;

zzN=0;
zz0=1;
%% Loop (only one loop!)
for zz=1:nMC
    zzN=zzN+1;
    
    
    % loading the dataset to be considered in RL-ABC
    if changedataset==1
        if dataset3==0 % consider only the two first datasets
            if zz<nMC/2
                if contdataset1==0 % load dataset1
                    dataset=dataset1;
                    % load experiment
                    run_dataset
                    contdataset1=1;
                else % do not load
                end
            else
                if contdataset2==0 % load dataset1
                    dataset=dataset2;
                    % load experiment
                    run_dataset
                    contdataset2=1;
                else % do not load
                end
            end
        else
            if zz<nMC/3
                if contdataset1==0 % load dataset1
                    dataset=dataset1;
                    % load experiment
                    run_dataset
                    contdataset1=1;
                else % do not load
                end
            elseif zz<nMC*2/3
                if contdataset2==0 % load dataset1
                    dataset=dataset2;
                    % load experiment
                    run_dataset
                    contdataset2=1;
                else % do not load
                end
            else
                if contdataset3==0 % load dataset3
                    dataset=dataset3;
                    % load experiment
                    run_dataset
                    contdataset3=1;
                else % do not load
                end
            end
        end
        
    end


    % Thompson Sampling, sample from beta_
    theta1=betarnd(alpha_(1),beta_(1));
    theta2=betarnd(alpha_(2),beta_(2));
    theta3=betarnd(alpha_(3),beta_(3));
    theta=[theta1;theta2;theta3];

    % select model with the biggest theta
    [Max,I] = max(theta);
    
    % not letting a model die out!
    [BCont,ImaxCont] = max(cont); %best model
    [WCont,IminCont] = min(cont); %worst model
    prctileMinB = prctile(betarnd(alpha_cont(ImaxCont),beta_cont(ImaxCont),10000,1),10);%  5, 10 or 15% percentile
    prctileMaxW = prctile(betarnd(alpha_cont(IminCont),beta_cont(IminCont),10000,1),90);%  95, 90 or 85% percentile
    Delta(zz)=prctileMaxW-prctileMinB;  %if Delta<0 stop rewarding
    if stoprewarding==0
        Delta(zz)=1; % keeping Delta>0
    end
   
    % pull this arm and collect reward
    if I==1
        
        % contador
        cont(I)=cont(I)+1;

        %random generator
        for ii=1:length(modparRef1)
            if modparRef1(ii)>0
                modpar(ii)=unifrnd(modparRef1_ll(ii),modparRef1_uu(ii));
            else % if mean negative, switch bounds
                modpar(ii)=unifrnd(modparRef1_uu(ii),modparRef1_ll(ii));
            end
        end
        
        % Deterministic model
        xbr=run_detmodel(modpar);
        
        % error
        errornow=norm(Xref-xbr)^2/norm(Xref)^2;

        if errornow<epsilon
            
            p_mod1=[p_mod1;modpar];
            xbr_mod1_ale=[xbr_mod1_ale;xbr];
    
            % contador
            contACC(I)=contACC(I)+1;
            globalACC(zz)=1;

            if errornow<erromin(I)
                erromin(I) = errornow
                p_mod1_id = modpar;
                xbr_mod1_id = xbr;
                ybr_mod1_id = yd;
            end
            
            if Delta(zz)<0
                % stop rewarding
                reward(zz,I)=0;
                % keep tracking of the reward history
                r=1; 
                reward_history_m1(contr1)=r;
                contr1=contr1+1;
                %
                alpha_cont(I)=alpha_cont(I)+r;
                beta_cont(I)=beta_cont(I)+(1-r);
            else
                r=1;
                reward(zz,I)=r;
                % keep tracking of the reward history
                reward_history_m1(contr1)=r;
                contr1=contr1+1;
                % short memory effect: considering only the last 'memory' points
                nn=length(reward_history_m1);
                if nn > memory
                    alpha_(I)= 2 + sum(reward_history_m1(nn-memory:nn));
                    beta_(I)= 2 + (memory-sum(reward_history_m1(nn-memory:nn)));
                    alpha_cont(I)= 2 + sum(reward_history_m1(nn-memory:nn));
                    beta_cont(I)= 2 + (memory-sum(reward_history_m1(nn-memory:nn)));
                else
                    alpha_(I)=alpha_(I)+r;
                    beta_(I)=beta_(I)+(1-r);
                    alpha_cont(I)=alpha_cont(I)+r;
                    beta_cont(I)=beta_cont(I)+(1-r);
                end
            end
        else
            globalACC(zz)=0;
            if Delta(zz)<0
                % stop rewarding
                reward(zz,I)=0;
                % keep tracking of the reward history
                r=0; 
                reward_history_m1(contr1)=r;
                contr1=contr1+1;
                %
                alpha_cont(I)=alpha_cont(I)+r;
                beta_cont(I)=beta_cont(I)+(1-r);
            else
                r=0;
                % keep tracking of the reward history
                reward_history_m1(contr1)=r;
                contr1=contr1+1;
                % short memory effect: considering only the last 'memory' points
                nn=length(reward_history_m1);
                if nn > memory
                    alpha_(I)= 2 + sum(reward_history_m1(nn-memory:nn));
                    beta_(I)= 2 + (memory-sum(reward_history_m1(nn-memory:nn)));
                    alpha_cont(I)= 2 + sum(reward_history_m1(nn-memory:nn));
                    beta_cont(I)= 2 + (memory-sum(reward_history_m1(nn-memory:nn)));
                else
                    alpha_(I)=alpha_(I)+r;
                    beta_(I)=beta_(I)+(1-r);
                    alpha_cont(I)=alpha_cont(I)+r;
                    beta_cont(I)=beta_cont(I)+(1-r);
                end
            end
        end

        
    elseif I==2
        
        % contador
        cont(I)=cont(I)+1;

        %random generator
        for ii=1:length(modparRef2)
            if modparRef2(ii)>0
                modpar(ii)=unifrnd(modparRef2_ll(ii),modparRef2_uu(ii));
            else % if mean negative, switch bounds
                modpar(ii)=unifrnd(modparRef2_uu(ii),modparRef2_ll(ii));
            end
        end
        
        
        % Deterministic model
        xbr=run_detmodel(modpar);
        
        % error
        errornow=norm(Xref-xbr)^2/norm(Xref)^2;

        if errornow<epsilon
            
            p_mod2=[p_mod2;modpar];
            xbr_mod2_ale=[xbr_mod2_ale;xbr];

            % contador
            contACC(I)=contACC(I)+1;
            globalACC(zz)=1;

            if errornow<erromin(I)
                erromin(I) = errornow
                p_mod2_id = modpar;
                xbr_mod2_id = xbr;
                ybr_mod2_id = yd;
            end
            
            if Delta(zz)<0
                % stop rewarding
                reward(zz,I)=0;
                % keep tracking of the reward history
                r=1; 
                reward_history_m2(contr2)=r;
                contr2=contr2+1;
                %
                alpha_cont(I)=alpha_cont(I)+r;
                beta_cont(I)=beta_cont(I)+(1-r);
            else
                r=1;
                reward(zz,I)=r;
                % keep tracking of the reward history
                reward_history_m2(contr2)=r;
                contr2=contr2+1;
                % short memory effect: considering only the last 'memory' points
                nn=length(reward_history_m2);
                if nn > memory
                    alpha_(I)= 2 + sum(reward_history_m2(nn-memory:nn));
                    beta_(I)= 2 + (memory-sum(reward_history_m2(nn-memory:nn)));
                    alpha_cont(I)= 2 + sum(reward_history_m2(nn-memory:nn));
                    beta_cont(I)= 2 + (memory-sum(reward_history_m2(nn-memory:nn)));
                else
                    alpha_(I)=alpha_(I)+r;
                    beta_(I)=beta_(I)+(1-r);
                    alpha_cont(I)=alpha_cont(I)+r;
                    beta_cont(I)=beta_cont(I)+(1-r);
                end
            end
        else
            globalACC(zz)=0;
            if Delta(zz)<0
                % stop rewarding
                reward(zz,I)=0;
                % keep tracking of the reward history
                r=0; 
                reward_history_m2(contr2)=r;
                contr2=contr2+1;
                %
                alpha_cont(I)=alpha_cont(I)+r;
                beta_cont(I)=beta_cont(I)+(1-r);
            else
                r=0;
                % keep tracking of the reward history
                reward_history_m2(contr2)=r;
                contr2=contr2+1;
                % short memory effect: considering only the last 'memory' points
                nn=length(reward_history_m2);
                if nn > memory
                    alpha_(I)= 2 + sum(reward_history_m2(nn-memory:nn));
                    beta_(I)= 2 + (memory-sum(reward_history_m2(nn-memory:nn)));
                    alpha_cont(I)= 2 + sum(reward_history_m2(nn-memory:nn));
                    beta_cont(I)= 2 + (memory-sum(reward_history_m2(nn-memory:nn)));
                else
                    alpha_(I)=alpha_(I)+r;
                    beta_(I)=beta_(I)+(1-r);
                    alpha_cont(I)=alpha_cont(I)+r;
                    beta_cont(I)=beta_cont(I)+(1-r);
                end
            end
        end
        
 elseif I==3
        
        % contador
        cont(I)=cont(I)+1;

        %random generator
        for ii=1:length(modparRef3)
            if modparRef3(ii)>0
                modpar(ii)=unifrnd(modparRef3_ll(ii),modparRef3_uu(ii));
            else
                modpar(ii)=unifrnd(modparRef3_uu(ii),modparRef3_ll(ii));
            end
        end
        
        % Deterministic model
        xbr=run_detmodel(modpar);
        
        % error
        errornow=norm(Xref-xbr)^2/norm(Xref)^2;

        if errornow<epsilon
            
            p_mod3=[p_mod3;modpar];
            xbr_mod3_ale=[xbr_mod3_ale;xbr];

            % contador
            contACC(I)=contACC(I)+1;
            globalACC(zz)=1;
           

            if errornow<erromin(I)
                erromin(I) = errornow
                p_mod3_id = modpar;
                xbr_mod3_id = xbr;
                ybr_mod3_id = yd;
            end

            if Delta(zz)<0
                % stop rewarding
                reward(zz,I)=0;
                % keep tracking of the reward history
                r=1; 
                reward_history_m3(contr3)=r;
                contr3=contr3+1;
                %
                alpha_cont(I)=alpha_cont(I)+r;
                beta_cont(I)=beta_cont(I)+(1-r);
            else
                r=1;
                reward(zz,I)=r;
                % keep tracking of the reward history
                reward_history_m3(contr3)=r;
                contr3=contr3+1;
                % short memory effect: considering only the last 'memory' points
                nn=length(reward_history_m3);
                if nn > memory
                    alpha_(I)= 2 + sum(reward_history_m3(nn-memory:nn));
                    beta_(I)= 2 + (memory-sum(reward_history_m3(nn-memory:nn)));
                    alpha_cont(I)= 2 + sum(reward_history_m3(nn-memory:nn));
                    beta_cont(I)= 2 + (memory-sum(reward_history_m3(nn-memory:nn)));
                else
                    alpha_(I)=alpha_(I)+r;
                    beta_(I)=beta_(I)+(1-r);
                    alpha_cont(I)=alpha_cont(I)+r;
                    beta_cont(I)=beta_cont(I)+(1-r);
                end
            end
        else
            globalACC(zz)=0;
            if Delta(zz)<0
                % stop rewarding
                reward(zz,I)=0;
                % keep tracking of the reward history
                r=0; 
                reward_history_m3(contr3)=r;
                contr3=contr3+1;
                %
                alpha_cont(I)=alpha_cont(I)+r;
                beta_cont(I)=beta_cont(I)+(1-r);
            else
                r=0;
                % keep tracking of the reward history
                reward_history_m3(contr3)=r;
                contr3=contr3+1;
                % short memory effect: considering only the last 'memory' points
                nn=length(reward_history_m3);
                if nn > memory
                    alpha_(I)= 2 + sum(reward_history_m3(nn-memory:nn));
                    beta_(I)= 2 + (memory-sum(reward_history_m3(nn-memory:nn)));
                    alpha_cont(I)= 2 + sum(reward_history_m3(nn-memory:nn));
                    beta_cont(I)= 2 + (memory-sum(reward_history_m3(nn-memory:nn)));
                else
                    alpha_(I)=alpha_(I)+r;
                    beta_(I)=beta_(I)+(1-r);
                    alpha_cont(I)=alpha_cont(I)+r;
                    beta_cont(I)=beta_cont(I)+(1-r);
               end
            end
        end
        
        
    end

    P1(zz)=sum(reward(zz0:zz,1))/zzN;
    P2(zz)=sum(reward(zz0:zz,2))/zzN;
    P3(zz)=sum(reward(zz0:zz,3))/zzN;

    % if not rewarding reset the model
    if zz>75
        if sum(globalACC(zz-75:zz))==0
            disp('Reseting mean model!!')
            % run identification
            run_identification
            modparRef1 = modpar_id1
            modparRef2 = modpar_id2
            modparRef3 = modpar_id3
            modparRef1_ll=modparRef1*lbound;
            modparRef1_uu=modparRef1*ubound;
            modparRef2_ll=modparRef2*lbound;
            modparRef2_uu=modparRef2*ubound;
            modparRef3_ll=modparRef3*lbound;
            modparRef3_uu=modparRef3*ubound;
            % resetting Beta distribution parameters
            alpha_=2*ones(1,M);
            beta_=2*ones(1,M);
            alpha_cont=2*ones(1,M);
            beta_cont=2*ones(1,M);
            zzN=0;
            zz0=zz;
            % resetting history
            clear reward_history_m1 reward_history_m2 reward_history_m3
            contr1=1; contr2=1; contr3=1;
            
        end
    end

end

%% Postprocessing

run_postproc


