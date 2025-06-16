%% Example 4.4

% This script simulates the SPDDE of Example 4.4 from the paper submission
% Magnus Methods for Stochastic Delay-Differential Equations.

%% USER PARAMETERS:
savefigures = 0; % Set this to 1 to save figures, and 0 when not saving figures.
runs = 1000;     % Number of trajectories.


%% SIMULATION BEGINNING:

for simulation = 0:2        % Simulation 0 is the deterministic case, 1 is uncorrelated noise, and 2 is correlated noise.
    pause(1);close all

    if simulation == 0
        stochastic = 0;                 % This to 1 to do a stochastic simulation, while 0 makes it deterministic.
        includespatialcorrelation = 0;  % This to 1 to include the spatial correlation to simulate W, while it is 0 if not using that.
    elseif simulation == 1
        stochastic = 1;
        includespatialcorrelation = 0;
    elseif simulation == 2
        stochastic = 1;
        includespatialcorrelation = 1;
    end

for delay = 0:1

%% PARAMETERS:

% SIMULATION OPTIONS:
referencesolution = 1; % Set this to 1 to simulate the reference solution with EM. Set it to 2 to use MEM.
showtitles = 0; % Show the titles on the graphs if this is 1, and suppress them if this is 0.

% Simulation parameters:
dtstart = 2^-0;
dtstartshowing = 2^-8;
dt = 2^-13;    % 10     % Smallest time increment in the approximations.
refdt = 2^-16; % 20     % Step size in reference solution.

if delay == 0
    runswas = runs;
    runs = 1;
else
    runs = runswas;
end
if simulation == 0
    runs = 1;
end

if runs == 1000     % This chooses a number of trials to use that ensures the figures are saved.
    savefigures = 1;
end

% Equation parameters:
ra = 1; rb = 10;
T0 = @(x) sin(2*pi*x);
d = 50;
m = d;
D = 1/25;
c = 0.15;
if simulation == 0
    c = 0;
end

Delx = 1/d;
x = (1:d)'*Delx;

U0 = zeros(d,1);
for k = 1:d
    U0(k) = T0(x(k));
end

A0 = zeros(d,d);
for row = 1:d
    if row == 1
        A0(row,row:row+1) = [-2,1];
    elseif row < d
        A0(row,row-1:row+1) = [1,-2,1];
    elseif row == d
        A0(row,:) = zeros(1,d);
    end
end
A0 = A0*D/(Delx^2);

A = zeros(d,d,m); % A_j as a dxd matrix, where the third component is the number j.
for j = 1:m
    A(:,:,j) = zeros(d,d);
    A(j,j,j) = c/sqrt(Delx);
end

% Cooling influence:
C1 = zeros(d,1); C2 = zeros(d,1);
for j = 1:d
    C1(j) = 1-d + j; C2(j) = 1-j;
end
C1 = C1*ra/(d-2); C2 = C2*rb/(d-2);
C1(end)=0;C2(end)=0;

f = @(t,Xd) C1*Xd(1) + C2*Xd(d-1);

T = 4;  % T > 1 is required in this script.

% Observation parameters:
hvalues = 2.^(log(dtstart)/log(2):-1:log(dt)/log(2));        % Set of step sizes to look at.
samplepath = 1;                             % Sample trajectory to view.
%samplecomponent = 1;                        % Sample component to view.
samplehvalue = dtstart*2^-6;                        % Sample step size to view.

%% CALCULATIONS BEGIN HERE.

if simulation == 0
    simulationtime = tic;
end

if delay == 0 && simulation == 0
    disp(['Value of D*dtstart/(Delx^2) is ',num2str(D*dtstart/(Delx^2)),', and value of D*refdt/(Delx^2) is ',num2str(D*refdt/(Delx^2)),'. For stable EM, this value must be less than 1/2.'])
    if D*dtstart/(Delx^2) >= 1/2 || D*refdt/(Delx^2) >= 1/2
        %error('Value of D*dtstart/(Delx^2) or D*refdt/(Delx^2) is too large.')
    end
end

%% INITIALISING CALCULATIONS:

rng('default');close all
T = ceil(T);    % This is ensuring T is an integer.

t = refdt:refdt:T;
tindex = (1:T)/refdt;

sumAj2s = zeros(d,d);
for j = 1:m
    sumAj2s = sumAj2s + A(:,:,j)^2;
end

A0hat = A0 - sumAj2s/2;

Urefvalues = zeros(d,T,runs);

% Initialise the errors, and compile them in vectors, so that we can view
% their means plotted against the step sizes, at the end of this script.
% The first variable refers to the time that the error is observed at. For
% example, ErrorEM(1,.,.) is the error at t = 1, while ErrorEM(2,.,.) is the
% error at t = 2.
ErrorEM = ones(T,runs,length(hvalues));     % Strong errors compared with X, a reference solution formed by the analytic expression with time step refdt.
ErrorMEM = ones(T,runs,length(hvalues));

% Stacks for the values:
EMvalues = zeros(d,T,runs,length(hvalues));  % Indexing is time, then trial, then step size for the scheme.
MEMvalues = zeros(d,T,runs,length(hvalues));

MSErrorEM = ones(T,length(hvalues));
MSErrorMEM = ones(T,length(hvalues));

%% SIMULATIONS BEGIN HERE:

for trial = 1:runs

%% SIMULATION OF WIENER PATH:

    % W and Wd, beginning with W:
        sqrtrefdt = sqrt(refdt);                    % Value used in the following calculation.
        dW = sqrtrefdt*randn(d,tindex(T));    % Brownian increments.  This is set up so that dW(1,n) = W(1,n+1) - W(1,n),  like DeltaW often is.
        dWinitial = sqrtrefdt*randn(d,1);
        
        Winitial = zeros(d,1);
        
        W = zeros(d,tindex(T));
        for n = 1:tindex(T)
            if n == 1
                W(:,n) = dWinitial(:,1);
            else
                W(:,n) = W(:,n-1) + dW(:,n-1);
            end
        end

        if trial == samplepath
            sampledW = dW;
            sampleW = W;
        end

        if includespatialcorrelation == 1
                        C = @(X,Y) min([X,Y]); % Covariance function.
                        % See page 37 of GhanemSpanos1991Book for the following results.
                        lambda = zeros(1,m);
                        for j = 1:m
                            lambda(j) = 4/(pi^2*(2*j-1)^2);
                        end
                        fn = zeros(d,length(lambda));
                        for i = 1:m
                            for j = 1:length(lambda)
                                fn(i,j) = sqrt(2)*sin(x(i)/sqrt(lambda(j)));    % fn(x(i),lambda(j))   %  fn_j(x_i)
                            end
                        end
                        
                        %KL = zeros(length(x),length(x),length(lambda)); % Karhunen--Loeve Approximation
                        KL = zeros(d,d); % Karhunen--Loeve Approximation
                        
                        k = 1;      % k = 1
                        for i = 1:d
                            for j = 1:d
                                %KL(i,j,k) = lambda(k)*fn(i,k)*fn(j,k);
                                KL(i,j) = lambda(k)*fn(i,k)*fn(j,k);
                            end
                        end
                        for k = 2:length(lambda)      % k = 2,3,...
                            for i = 1:d
                                for j = 1:d
                                    %KL(i,j,k) = KL(i,j,k-1) + lambda(k)*fn(i,k)*fn(j,k);
                                    KL(i,j) = KL(i,j) + lambda(k)*fn(i,k)*fn(j,k);
                                end
                            end
                        end

                        for j = 1:m
                            A(:,:,j) = zeros(d,d);
                            for i = 1:d
                                A(i,i,j) = c/sqrt(Delx)*sqrt(lambda(j))*fn(i,j);
                            end
                        end

                        sumAj2s = zeros(d,d);
                        for j = 1:m
                            sumAj2s = sumAj2s + A(:,:,j)^2;
                        end
                        
                        A0hat = A0 - sumAj2s/2;


                        xi = dW/sqrtrefdt; % The Brownian noise, which is dW but not scaled with time.


                        wcinitial = zeros(d,1);
                        wc = zeros(d,length(t));
                        for k = 1:length(lambda)      % k = 1,2,3,...
                            for i = 1:d
                                for n = 1:length(t)
                                    wc(i,n) = sqrt(lambda(k))*xi(i,n)*fn(i,k);
                                end
                            end
                        end
                        Wc = zeros(size(wc));
                        Wcinitial = zeros(d,1);
%                         beta = zeros(size(wc));
%                         betainitial = zeros(d,1);
%                         beta(:,1) = sqrt(refdt)*randn(d,1);
%                         for n = 1:length(t)-1
%                             beta(:,n+1) = sqrt(refdt)*xi(:,n)+beta(:,n);
%                         end
                        k = 1;
                        Wc(i,n) = sqrt(lambda(k))*fn(i,k)*W(i,n);
                        for k = 2:length(lambda)      % k = 2,3,...
                            for i = 1:d
                                for n = 1:length(t)
                                    Wc(i,n) = Wc(i,n) + sqrt(lambda(k))*fn(i,k)*W(i,n);
                                end
                            end
                        end

                        
                        if trial == samplepath
                            sampledW = wc;
                            sampleW = Wc;
                        end
                        
        end

        

        % Define Wd(t) = W(t-1):
        Wd = zeros(d,tindex(T));
        dWd = zeros(d,tindex(T));
        Wd(:,tindex(1)) = Winitial(:,1);
        for i = 1:tindex(T-1)
            Wd(:,tindex(1)+i) = W(:,i);
        end
        for n = tindex(1):tindex(T)-1
            dWd(:,n) = Wd(:,n+1)-Wd(:,n);
        end

        
        
%% SIMULATION OF REFERENCE SOLUTION:

    % Construct the reference (analytic) solution (with step size h = refdt):
    Uref = zeros(d,tindex(T));
    if referencesolution == 1
        % Initial step:
        tn = 0; Un = U0; Ud = U0;
        a = A0*Un + f(tn,Ud);
        bsum = zeros(d,1); for j = 1:m; bsum = bsum + A(:,:,j)*Un*dWinitial(j); end
        Uref(:,1) = Un + a*refdt + bsum;
        % First time interval until first delay point:
        for n = 1:tindex(1)-1
            Un = Uref(:,n); Ud = U0;
            a = A0*Un + f(tn,Ud);
            bsum = zeros(d,1); for j = 1:m; bsum = bsum + A(:,:,j)*Un*dW(j,n); end
            if delay == 0
                a = A0*Un + f(tn,Un);
            end
            Uref(:,n+1) = Un + a*refdt + bsum;
        end
        % Successive intervals between delays:
        for Ti = 2:T
            for n = tindex(Ti-1):tindex(Ti)-1
                Un = Uref(:,n);
                if n == tindex(1)
                    Ud = U0;
                else
                    Ud = Uref(:,n-tindex(1));
                end
                a = A0*Un + f(tn,Ud);
                bsum = zeros(d,1); for j = 1:m; bsum = bsum + A(:,:,j)*Un*dW(j,n); end
                if delay == 0
                    a = A0*Un + f(tn,Un);
                end
                Uref(:,n+1) = Un + a*refdt + bsum;
            end
        end
    elseif referencesolution == 2
        % Initial step:
        tn = 0; Un = U0; Ud = U0;
        a = f(tn,Ud);
        bsum = zeros(d,1); for j = 1:m; bsum = bsum + A(:,:,j)*dWinitial(j); end
        expOmega1 = expm( A0hat*refdt + bsum );
        Uref(:,1) = expOmega1*(Un+a*refdt);
        % Iterative steps:
        for n = 1:length(t)-1
                Un = Uref(:,n);
                if n <= tindex(1)
                    Ud = U0;
                else
                    Ud = Uref(:,n-tindex(1));
                end
                a = f(tn,Ud);
                bsum = zeros(d,1); for j = 1:m; bsum = bsum + A(:,:,j)*dW(j,n); end
                if delay == 0
                    a = f(tn,Un);
                end
                expOmega1 = expm( A0hat*refdt + bsum );
                Uref(:,n+1) = expOmega1*(Un+a*refdt);
        end
    end
    %Xref = Xref(:,1:end-1); A condition such as this is needed it we don't have the calculation as we do at (*), just above, and allow Xref to be defined beyond T.
    for ti = 1:T
        Urefvalues(:,ti,trial) = Uref(:,ti/refdt);
    end
    if trial == samplepath
        sampleUref = Uref(:,:);
    end
    if delay == 0
        sampleDelayFreeUref = sampleUref;
    end
        
%% SCHEMES:

if delay == 1

    for h = 1:length(hvalues)

        % Define some constants that are used throughout the following part:
        step = hvalues(h)/refdt;    % This is the index multiplier, for how many index positions comprise a single step of the simulations.  The idea is that t_n = t(n*step).
        N = tindex(T)/step;           % This is the number of steps that are made in [0,T].          
        Delt = hvalues(h);          % Step size.                 
        A0hatDelt = A0hat*Delt;

        % DelW1 and DelW1d:
        DelW = zeros(d,N-1);  % Define Delta W1 (see page 340 of Kloeden and Platen).
        DelWinitial = zeros(d,1);
        DelWd = zeros(d,N-1);
        DelWdinitial = zeros(d,1);
        DelWinitial(:,1) = W(:,step);
        for n = 1:N-1
            DelW(:,n) = W(:,(n+1)*step) - W(:,n*step);
            if n == N/T
                DelWdinitial(:,1) = DelWinitial(:,1);
                DelWd(:,n) = DelWdinitial(:,1);
            elseif n > N/T
                DelWd(:,n) = DelW(:,n-N/T);
            end
        end

        % Initialise the schemes:
        EM = zeros(d,N);       % Euler--Maruyama 
        MEM = zeros(d,N);      % Magnus--EM

        % The first step, from t=0 to t=h, using initial increments, such
        % as DelW1initial.  Effectively, this is n = 0:

        % EM:
        tn = 0; Yn = U0; Yd = U0;
        a = A0*Yn + f(tn,Yd);
        bsum = zeros(d,1); for j = 1:m; bsum = bsum + A(:,:,j)*Yn*DelWinitial(j,1); end
        EM(:,1) = Yn + a*Delt + bsum;

        % MEM:
        tn = 0; Yn = U0; Yd = U0;
        a = f(tn,Yd);
        bsum = zeros(d,1); for j = 1:m; bsum = bsum + A(:,:,j)*DelWinitial(j,1); end
        expOmega1 = expm( A0hatDelt + bsum );
        MEM(:,1) = expOmega1*(Yn + a*Delt );
        

%% STEP ONE:

        % Beginning of scheme calculation from time h to tau.
        for n = 1 : N/T - 1
            
            % EM:
            tn = n*step; Yn = EM(:,n); Yd = U0;
            a = A0*Yn + f(tn,Yd);
            bsum = zeros(d,1); for j = 1:m; bsum = bsum + A(:,:,j)*Yn*DelW(j,n); end
            EM(:,n+1) = Yn + a*Delt + bsum;
             
            % MEM:
            tn = n*step; Yn = MEM(:,n); Yd = U0;
            a = f(tn,Yd);
            bsum = zeros(d,1); for j = 1:m; bsum = bsum + A(:,:,j)*DelW(j,n); end
            expOmega1 = expm( A0hatDelt + bsum );
            MEM(:,n+1) = expOmega1*(Yn + a*Delt );
            
        end
        % End of scheme calculation from time h to tau.

%% STEP TWO:

        % Beginning for time tau <= t < 2*tau.
        for n = N/T:2*N/T-1
            
            % EM:
            tn = n*step; Yn = EM(:,n);
            if n == N/T
                Yd = U0;
            else
                Yd = EM(:,n-N/T);
            end       
            a = A0*Yn + f(tn,Yd);
            bsum = zeros(d,1); for j = 1:m; bsum = bsum + A(:,:,j)*Yn*DelW(j,n); end
            EM(:,n+1) = Yn + a*Delt + bsum;
         
            % MEM:
            tn = n*step; Yn = MEM(:,n); 
            if n == N/T
                Yd = U0;
            else
                Yd = MEM(:,n-N/T);
            end 
            a = f(tn,Yd);
            bsum = zeros(d,1); for j = 1:m; bsum = bsum + A(:,:,j)*DelW(j,n); end
            expOmega1 = expm( A0hatDelt + bsum );
            MEM(:,n+1) = expOmega1*(Yn + a*Delt);
            
        end
        % End for time tau <= t < 2*tau.

%% STEP THREE:

        % Beginning for time 2*tau <= t < 3*tau.
        if T > 2
            for n = 2*N/T : 3*N/T-1
                
                % EM:
                tn = n*step; Yn = EM(:,n);
                Yd = EM(:,n-N/T); 
                a = A0*Yn + f(tn,Yd);
                bsum = zeros(d,1); for j = 1:m; bsum = bsum + A(:,:,j)*Yn*DelW(j,n); end
                EM(:,n+1) = Yn + a*Delt + bsum;
              
                % MEM:
                tn = n*step; Yn = MEM(:,n); 
                    Yd = MEM(:,n-N/T);
                a = f(tn,Yd);
                bsum = zeros(d,1); for j = 1:m; bsum = bsum + A(:,:,j)*DelW(j,n); end
                expOmega1 = expm( A0hatDelt + bsum );
                MEM(:,n+1) = expOmega1*(Yn + a*Delt);
                      
                
            end
        end
        % End for time 2*tau <= t < 3*tau.

%% STEPS FOUR ONWARDS

        % Beginning for time (Ti-1)*tau <= t < Ti*tau, for Ti >= 4.
        if T >= 4
            for Ti = 4:T
                for n = (Ti-1)*N/T : Ti*N/T-1
                    
                    % Schemes on these steps:
    
                    % EM:
                    tn = n*step; Yn = EM(:,n);
                    Yd = EM(:,n-N/T); 
                    a = A0*Yn + f(tn,Yd);
                    bsum = zeros(d,1); for j = 1:m; bsum = bsum + A(:,:,j)*Yn*DelW(j,n); end
                    EM(:,n+1) = Yn + a*Delt + bsum;
                                    
                    % MEM:
                    tn = n*step; Yn = MEM(:,n); 
                        Yd = MEM(:,n-N/T);
                    a = f(tn,Yd);
                    bsum = zeros(d,1); for j = 1:m; bsum = bsum + A(:,:,j)*DelW(j,n); end
                    expOmega1 = expm( A0hatDelt + bsum );
                    MEM(:,n+1) = expOmega1*(Yn + a*Delt);

                end
            end
        end
        % End for time (Ti-1)*tau <= t < Ti*tau, for Ti >= 4.

%% SAVING VALUES:        

        if trial == samplepath && hvalues(h) == samplehvalue
            sampletime = t(step:step:end);
            sampleEM = EM(:,:);
            sampleMEM = MEM(:,:);  
        end

        for ti = 1:T
            EMvalues(:,ti,trial,h) = EM(:,ti*N/T);
            MEMvalues(:,ti,trial,h) = MEM(:,ti*N/T);
        end

        for ti = 1:T
            ErrorEM(ti,trial,h) = norm(EMvalues(:,ti,trial,h)-Urefvalues(:,ti,trial));
                        %temporary = MEMvalues(end-1,ti,trial,h) - Urefvalues(end-1,ti,trial);
            ErrorMEM(ti,trial,h) = norm(MEMvalues(:,ti,trial,h)-Urefvalues(:,ti,trial));
                        %ErrorMEM(ti,trial,h) = norm(temporary);
        end

    end

end

end



end

%% ERROR CALCULATIONS

for ti = 1:T
    for h = 1:length(hvalues)
        MSErrorEM(ti,h) = sqrt( mean( squeeze( ErrorEM(ti,:,h).^2 ) ) );
        MSErrorMEM(ti,h) = sqrt( mean( squeeze( ErrorMEM(ti,:,h).^2 ) ) );
    end
end

%% BROWNIAN SHEET:

figure(T+1)
hold on
[Zx,Zy] = meshgrid([0,t],x);
Z = zeros(size(Zx));
for space = 1:d
    for time = 1:length(t)+1
        if time == 1
            Z(space,time) = Winitial(space);
        else
            Z(space,time) = sampleW(space,time-1);
        end
    end
end
surf(Zx,Zy,Z,'EdgeColor','none','FaceColor','interp')
xlabel('$t$','Interpreter','latex','FontSize',16)
ylabel('$x$','Interpreter','latex','FontSize',16)
zlabel('$W(t,x)$','Interpreter','latex','FontSize',16)
if includespatialcorrelation == 1
    zlabel('$W^c(t,x)$','Interpreter','latex','FontSize',16)
end
%zticks('')
view(0,90)
colorbar
axis([0,T,x(1),x(end)])
set(gcf,'position',[200,75,700,500])
if includespatialcorrelation == 0
    if savefigures == 1
        exportgraphics(gcf,'ZExample4UncorrelatedQWiener.png','Resolution',300)
    end
     if showtitles == 1
        title('Heatmap of $W$','Interpreter','latex','FontSize',16)
    end
else
    if savefigures == 1
        exportgraphics(gcf,'ZExample4CorrelatedQWiener.png','Resolution',300)
    end
    if showtitles == 1
        title('Heatmap of $W^c$','Interpreter','latex','FontSize',16)
    end
end
view(0,90)

%% TRAJECTORY PLOT:

figure(T+2)
hold on
[Zx,Zy] = meshgrid([0,t],[0;x]);
Z = zeros(size(Zx));
for space = 1:d
    for time = 1:length(t)+1
        if time == 1
            Z(space+1,time) = U0(space);
        else
            Z(space+1,time) = sampleUref(space,time-1);
        end
    end
end

surf(Zx,Zy,Z,'EdgeColor','none','FaceColor','interp')
hold on
plot3(zeros(length(x)+1,1),[0;x],[0;U0(:)],'k','LineWidth',1)
plot3(T*ones(length(x),1),x,sampleUref(:,end),'k','LineWidth',1)
plot3(t,zeros(length(t),1),zeros(length(t),1),'k','LineWidth',1)
for j = 5:5:d
    plot3(t,x(j)*ones(length(t),1),sampleUref(j,:),'k','LineWidth',1)
end
for time = T/16:T/16:T
    i = find(round(t,10)==time,1);
    plot3(t(i)*ones(d+1,1),[0;x],[0;sampleUref(:,i)],'k','LineWidth',1)
end
xlabel('$t$','Interpreter','latex','FontSize',16)
ylabel('$x$','Interpreter','latex','FontSize',16)
zlabel('$U(t,x)$','Interpreter','latex','FontSize',16)
%zticks('')
view(-60,45)
view(-30,15)
axis([0,T,0,x(end)])
set(gcf,'position',[200,75,700,500])
if ~isempty(find(round(x,5)==0.7,1))
    j = find(round(x,5)==0.7);
    plot3([0,t],x(j)*ones(length(t)+1,1),[U0(j),sampleUref(j,:)],'r--','LineWidth',3)
end
if ~isempty(find(round(t,10)==1.25,1))
    n = find(round(t,10)==1.25);
    plot3(t(n)*ones(length(x)+1,1),[0;x],[0;sampleUref(:,n)],'r--','LineWidth',3)
end
if includespatialcorrelation == 0
    if savefigures == 1
        exportgraphics(gcf,'ZExample4UncorrelatedSamplePathGeneralView.png','Resolution',300)
    end
    if showtitles == 1
        title('Heatmap of $U$ with Uncorrelated Noise','Interpreter','latex','FontSize',16)
    end
else
    if savefigures == 1
        exportgraphics(gcf,'ZExample4CorrelatedSamplePathGeneralView.png','Resolution',300)
    end
    if showtitles == 1
        title('Heatmap of $U$ with Correlated Noise','Interpreter','latex','FontSize',16)
    end
end
if simulation == 0
    if savefigures == 1
        exportgraphics(gcf,'ZExample4DeterministicSamplePathGeneralView.png','Resolution',300)
    end
end
view(0,90)
colorbar
if includespatialcorrelation == 0
    if savefigures == 1
        exportgraphics(gcf,'ZExample4UncorrelatedSamplePath.png','Resolution',300)
    end
    if showtitles == 1
        title('Heatmap of $U$ with Uncorrelated Noise','Interpreter','latex','FontSize',16)
    end
else
    if savefigures == 1
        exportgraphics(gcf,'ZExample4CorrelatedSamplePath.png','Resolution',300)
    end
    if showtitles == 1
        title('Heatmap of $U$ with Correlated Noise','Interpreter','latex','FontSize',16)
    end
end
if simulation == 0
    if savefigures == 1
        exportgraphics(gcf,'ZExample4DeterministicSamplePath.png','Resolution',300)
    end
end

%% ERROR GRAPHS:

% Error graph (t=Ti), for Ti = 1,2,...,T:
for Ti = T
    figure(Ti)
    hold on
    referenceorderhalf = MSErrorMEM(Ti,-log(dtstartshowing)/log(2)) *sqrt(hvalues);
    referenceorderone = min( [MSErrorMEM(Ti,-log(dtstartshowing)/log(2)),MSErrorMEM(Ti,-log(dtstartshowing)/log(2))] )*hvalues;
    plot(log(hvalues)/log(2),log10(referenceorderhalf)+1.15,'m-o','Linewidth',1)
    plot(log(hvalues)/log(2),log10(referenceorderone)+1.2,'m-square','Linewidth',1)
    plot(log(hvalues)/log(2),log(MSErrorEM(Ti,:))/log(10),'r--o','Linewidth',1)
    plot(log(hvalues)/log(2),log(MSErrorMEM(Ti,:))/log(10),'-o','color',[1,1/2,0],'Linewidth',1)
    xticks(log(dt)/log(2):0)
    xlabel('$\log_2h$','Interpreter','latex','FontSize',16)
        legend('Reference $1/2$','Reference $1$','EM','MEM','Interpreter','latex','FontSize',14,'location','Southeast','NumColumns',1)
    xlim([log(dt)/log(2),log(dtstartshowing)/log(2)])
    ylabel('$\log_{10}\mathrm{MSE}(4)$','Interpreter','latex','FontSize',16)
    %yticks(-10:0.5:10)
    %ylim([-4,-2])
    if showtitles == 1
        title(['MS Strong Errors ','($t=$ ',num2str(Ti),')'],'Interpreter','latex','FontSize',16)
    end
    set(gcf,'position',[200,75,700,500])
    grid on
    a = get(gca,'XTickLabel');
    set(gca,'XTickLabel',a,'fontsize',12)
    b = get(gca,'YTickLabel');
    set(gca,'YTickLabel',b,'fontsize',12)
            if Ti == 4
                if includespatialcorrelation == 0
                    if savefigures == 1
                        saveas(gcf,'ZExample4EGsUncorrelatedZoomed.eps','epsc')
                    end
                else
                    if savefigures == 1
                        saveas(gcf,'ZExample4EGsCorrelatedZoomed.eps','epsc')
                    end
                end
            end
end


% SAMPLE FULL EG:
Ti = 4;
figure(T+3)
hold on
referenceorderhalf = max( [MSErrorEM(Ti,1),MSErrorMEM(Ti,1)] )*sqrt(hvalues);
referenceorderone = max( [MSErrorEM(Ti,1),MSErrorMEM(Ti,1)] )*hvalues;
%plot(log(hvalues)/log(2),log10(referenceorderhalf),'m-o','Linewidth',1)
%plot(log(hvalues)/log(2),log10(referenceorderone),'m-square','Linewidth',1)
plot(log(hvalues)/log(2),log(MSErrorEM(Ti,:))/log(10),'r--o','Linewidth',1)
plot(log(hvalues)/log(2),log(MSErrorMEM(Ti,:))/log(10),'-o','color',[1,1/2,0],'Linewidth',1)
xticks(log(dt)/log(2):0)
xlabel('$\log_2h$','Interpreter','latex','FontSize',16)
    %legend('Reference $1/2$','Reference $1$','EM','MEM','Interpreter','latex','FontSize',14,'location','Northwest','NumColumns',1)
legend('EM','MEM','Interpreter','latex','FontSize',14,'location','Northwest','NumColumns',1)
xlim([log(dt)/log(2),log(dtstart)/log(2)])
ylabel('$\log_{10}\mathrm{MSE}(4)$','Interpreter','latex','FontSize',16)
%title(['MS Strong Errors ','($t=$ ',num2str(Ti),')'],'Interpreter','latex','FontSize',16)
set(gcf,'position',[200,75,700,500])
grid on
a = get(gca,'XTickLabel');
set(gca,'XTickLabel',a,'fontsize',12)
b = get(gca,'YTickLabel');
set(gca,'YTickLabel',b,'fontsize',12)
        if Ti == 4
            if savefigures == 1
                if includespatialcorrelation == 0
                    saveas(gcf,'ZExample4EGsUncorrelatedFull.eps','epsc')
                else
                    saveas(gcf,'ZExample4EGsCorrelatedFull.eps','epsc')
                end
            end
        end

%% CROSS SECTIONS:

if ~isempty(find(round(x,5)==0.7,1))
    figure(T+4)
    hold on
    j = find(round(x,5)==0.7,1);
    plot(t,sampleUref(j,:),'r','LineWidth',2);
    plot(t,sampleDelayFreeUref(j,:),'Color',[0,0,2/3],'LineWidth',2);
    legend('$\tau=1$','$\tau=0$','Interpreter','latex','FontSize',16,'Location','Northeast')
    xlabel('$t$','Interpreter','latex','FontSize',16)
    ylabel('$U(t,0.7)$','Interpreter','latex','FontSize',16)
    xlim([0,T])
    ylim([-1,1])
    yticks(-10:0.5:10)
    %axis([0,T,x(1),x(end)])
    set(gcf,'position',[200,75,700,500])
    if showtitles == 1
        title('Cross Section ($x=0.7$)','Interpreter','latex','FontSize',16)
    end
    if savefigures == 1
        if includespatialcorrelation == 0
            saveas(gcf,'ZExample4CrossUncorrelated.eps','epsc')
        else
            saveas(gcf,'ZExample4CrossCorrelated.eps','epsc')
        end
        if simulation == 0
            saveas(gcf,'ZExample4CrossDeterministic.eps','epsc')
        end
    end
end

figure(T+5)
hold on
if ~isempty(find(round(t,10)==1.25,1))
    n = find(round(t,10)==1.25);
    plot([0;x],[0;sampleUref(:,n)],'r','LineWidth',2);
    plot([0;x],[0;sampleDelayFreeUref(:,n)],'Color',[0,0,2/3],'LineWidth',2);
    legend('$\tau=1$','$\tau=0$','Interpreter','latex','FontSize',16,'Location','Northeast')
    xlabel('$x$','Interpreter','latex','FontSize',16)
    ylabel('$U(1.25,x)$','Interpreter','latex','FontSize',16)
    xlim([0,1])
    ylim([-0.2,0.8])
    yticks(-0.2:0.1:0.8)
    %axis([0,T,x(1),x(end)])
    set(gcf,'position',[200,75,700,500])
    a = get(gca,'XTickLabel');
    set(gca,'XTickLabel',a,'fontsize',16)
    b = get(gca,'YTickLabel');
    set(gca,'YTickLabel',b,'fontsize',12)
    if showtitles == 1
        title('Cross Section ($t=1.25$)','Interpreter','latex','FontSize',16)
    end
    if savefigures == 1
        if includespatialcorrelation == 0
            saveas(gcf,'ZExample4CrossTimeUncorrelated.eps','epsc')
        else
            saveas(gcf,'ZExample4CrossTimeCorrelated.eps','epsc')
        end
        if simulation == 0
            saveas(gcf,'ZExample4CrossTimeDeterministic.eps','epsc')
        end
    end
end


%% TRAJECTORY WITH SCHEMES:

figure(T+6)
hold on
[Zx,Zy] = meshgrid([0,t],x);
Z = zeros(size(Zx));
for space = 1:d
    for time = 1:length(t)+1
        if time == 1
            Z(space,time) = U0(space);
        else
            Z(space,time) = sampleUref(space,time-1);
        end
    end
end

[Zxscheme,Zyscheme] = meshgrid([0,sampletime],x);
Zscheme = zeros(size(Zxscheme));
for space = 1:d
    for time = 1:length(sampletime)+1
        if time == 1
            Zscheme(space,time) = U0(space);
        else
            Zscheme(space,time) = sampleEM(space,time-1);
        end
    end
end

surf(Zx,Zy,Z,'EdgeColor','none','FaceColor','interp')
surf(Zxscheme,Zyscheme,Zscheme,'EdgeColor','none','FaceColor','r','FaceAlpha',0.1)
hold on
plot3(zeros(length(x),1),x,U0(:),'k','LineWidth',1)
plot3(T*ones(length(x),1),x,sampleUref(:,end),'k','LineWidth',1)
plot3(t,zeros(length(t),1),zeros(length(t),1),'k','LineWidth',1)
for j = 5:5:d
    plot3(t,x(j)*ones(length(t),1),sampleUref(j,:),'k','LineWidth',1)
end
for time = T/16:T/16:T
    i = find(round(t,10)==time,1);
    plot3(t(i)*ones(length(x),1),x,sampleUref(:,i),'k','LineWidth',1)
end
xlabel('$t$','Interpreter','latex','FontSize',16)
ylabel('$x$','Interpreter','latex','FontSize',16)
zlabel('$U(t,x)$','Interpreter','latex','FontSize',16)
%zticks('')
title('EM Scheme Alongside Solution','Interpreter','latex','FontSize',16)
view(-60,45)
view(-30,15)
axis([0,T,x(1),x(end)])
set(gcf,'position',[200,75,700,500])



figure(T+7)
hold on
[Zx,Zy] = meshgrid([0,t],x);
Z = zeros(size(Zx));
for space = 1:d
    for time = 1:length(t)+1
        if time == 1
            Z(space,time) = U0(space);
        else
            Z(space,time) = sampleUref(space,time-1);
        end
    end
end

[Zxscheme,Zyscheme] = meshgrid([0,sampletime],x);
Zscheme = zeros(size(Zxscheme));
for space = 1:d
    for time = 1:length(sampletime)+1
        if time == 1
            Zscheme(space,time) = U0(space);
        else
            Zscheme(space,time) = sampleMEM(space,time-1);
        end
    end
end

surf(Zx,Zy,Z,'EdgeColor','none','FaceColor','interp')
surf(Zxscheme,Zyscheme,Zscheme,'EdgeColor','none','FaceColor','g','FaceAlpha',0.1)
hold on
plot3(zeros(length(x),1),x,U0(:),'k','LineWidth',1)
plot3(T*ones(length(x),1),x,sampleUref(:,end),'k','LineWidth',1)
plot3(t,zeros(length(t),1),zeros(length(t),1),'k','LineWidth',1)
for j = 5:5:d
    plot3(t,x(j)*ones(length(t),1),sampleUref(j,:),'k','LineWidth',1)
end
for time = T/16:T/16:T
    i = find(round(t,10)==time,1);
    plot3(t(i)*ones(length(x),1),x,sampleUref(:,i),'k','LineWidth',1)
end
xlabel('$t$','Interpreter','latex','FontSize',16)
ylabel('$x$','Interpreter','latex','FontSize',16)
zlabel('$U(t,x)$','Interpreter','latex','FontSize',16)
%zticks('')
title('MEM Scheme Alongside Solution','Interpreter','latex','FontSize',16)
view(-60,45)
view(-30,15)
axis([0,T,x(1),x(end)])
set(gcf,'position',[200,75,700,500])

if ~isempty(find(round(x,5)==0.5,1))
    figure(T+8)
    hold on
    j = find(round(x,5)==0.5,1);
    plot([-1,0,t],[U0(j),U0(j),sampleUref(j,:)],'k','LineWidth',2);
    plot([-1,0,sampletime],[U0(j),U0(j),sampleEM(j,:)],'Color','r','LineWidth',1);
    plot([-1,0,sampletime],[U0(j),U0(j),sampleMEM(j,:)],'Color','g','LineWidth',1);
    legend('$U(t,0.7)$','EM','MEM','Interpreter','latex','FontSize',16,'Location','Northeast')
    xlabel('$t$','Interpreter','latex','FontSize',16)
    ylabel('$U(t,0.5)$','Interpreter','latex','FontSize',16)
    xlim([0,T])
    %yticks(-1:0.5:1)
    %axis([0,T,x(1),x(end)])
    set(gcf,'position',[200,75,700,500])
    title('Cross Section of Solution and Schemes','Interpreter','latex','FontSize',16)
end

end

disp(['Total simulation time is ',num2str(round(toc(simulationtime)/60,4)),' minutes.'])