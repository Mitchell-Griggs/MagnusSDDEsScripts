% This script is for plotting the mean-square error graphs of the semilinear,
% homogeneous, d-dimensional SDDE
%  dX(t) = \sum_{j=0}^m ( A_j*X(t) + B_j(X(t),X(t-d1),X(t-d2)) dW_j(t),
% 0<=t<=T, with X(t) = X(0)=X0 for t<=0,   where m = 2 and d = 2, but the
% Jacobians are manually calculated.

%% USER PARAMETERS:
saveplots = 0; % Set this to 1 to save figures, and 0 when not saving figures.
runs = 10000;  % Number of trajectories.

% Equation parameters:
X0 = [0.8;0.2];
A0 = [-0.1,0.03;-0.2,-0.04];   A1 = [0.15,0.1;0.2,0.1];  A2 = [0.05,0.03;0.04,0.01];

% For this semilinear script, to have the full nonlinearity, we need to
% define the derivative functions here, too, because the syms function does
% not well handlde Jacobian calculations.

B0 = @(X,Xd1,Xd2) [sin(X(1));cos(X(2))]/5; 
B1 = @(X,Xd1,Xd2) [Xd2(1)-Xd1(1);Xd1(2)-Xd2(2)]/10; 
B2 = @(X,Xd1,Xd2) [sin(X(2)*Xd1(2)*Xd2(2));cos(X(1)*Xd1(1)*Xd2(1))]/5;

B1x = @(X,Xd1,Xd2) [0,0;0,0]/10;
B1xd1 = @(X,Xd1,Xd2) [-1,0;0,1]/10;
B1xd2 = @(X,Xd1,Xd2) [1,0;0,-1]/10;
B2x = @(X,Xd1,Xd2) [0,Xd1(2)*Xd2(2)*cos(X(2)*Xd1(2)*Xd2(2));-Xd1(1)*Xd2(1)*sin(X(1)*Xd1(1)*Xd2(1)),0]/5;
B2xd1 = @(X,Xd1,Xd2) [0,X(2)*Xd2(2)*cos(X(2)*Xd1(2)*Xd2(2));-X(1)*Xd2(1)*sin(X(1)*Xd1(1)*Xd2(1)),0]/5;
B2xd2 = @(X,Xd1,Xd2) [0,X(2)*Xd1(2)*cos(X(2)*Xd1(2)*Xd2(2));-X(1)*Xd1(1)*sin(X(1)*Xd1(1)*Xd2(1)),0]/5;

%% SIMULATION BEGINNING:

useRefinedMilsteinschemes = 1; % Set this to 1 to simulate the refined Milstein schemes and the double integrals. This is significantly slower than without.

% Simulation parameters:
dt = 2^-10;    % 10     % Smallest time increment in the approximations.
refdt = 2^-14; % 20     % Step size in reference solution.

d1 = 1; d2 = 1/4;

T = 4;  % T > 1 is required in this script.
        
% Observation parameters:
showerrorplots = 1;
hvalues = 2.^(log(d2)/log(2):-1:log(dt)/log(2));        % Set of step sizes to look at.
samplepath = 1;                             % Sample trajectory to view.
samplecomponent = 1;                        % Sample component to view.
samplehvalue = 2^-2;                        % Sample step size to view.

%% CALCULATIONS BEGIN HERE.

simulationtime = tic;

% % Calculate Jacobians:
% syms x1 x2 y1 y2 z1 z2
%     B1xtemp = eval(['@(x1,x2)' char(jacobian(B1([x1;x2],[y1;y2],[z1;z2]),[x1;x2]))]);
%     B1x = @(X,Y,Z) B1xtemp(X(1),X(2));
%     B1xd1temp = eval(['@(y1,y2)' char(jacobian(B1([x1;x2],[y1;y2],[z1;z2]),[y1;y2]))]);
%     B1xd1 = @(X,Y,Z) B1xd1temp(Y(1),Y(2));
%     B1xd2temp = eval(['@(z1,z2)' char(jacobian(B1([x1;x2],[y1;y2],[z1;z2]),[z1;z2]))]);
%     B1xd2 = @(X,Y,Z) B1xd2temp(Z(1),Z(2));
%     B2xtemp = eval(['@(x1,x2)' char(jacobian(B2([x1;x2],[y1;y2],[z1;z2]),[x1;x2]))]);
%     B2x = @(X,Y,Z) B2xtemp(X(1),X(2));
%     B2xd1temp = eval(['@(y1,y2)' char(jacobian(B2([x1;x2],[y1;y2],[z1;z2]),[y1;y2]))]);
%     B2xd1 = @(X,Y,Z) B2xd1temp(Y(1),Y(2));
%     B2xd2temp = eval(['@(z1,z2)' char(jacobian(B2([x1;x2],[y1;y2],[z1;z2]),[z1;z2]))]);
%     B2xd2 = @(X,Y,Z) B2xd2temp(Z(1),Z(2));
% clear x1 x2 y1 y2 z1 z2

%% INITIALISING CALCULATIONS:

rng('default');close all
T = ceil(T);    % This is ensuring T is an integer.

d = 2;  % Dimension.
t = refdt:refdt:T;  % Time.
td1index = (d1:d1:T)/refdt; % Positions in t when t = k*d1 for some integer k.
td2index = (d2:d2:T)/refdt;   % Positions in t when t = k*d2 for some integer k.
% To test these, run the script and then put t(td1index) and t(td2index) into
% the command window.
tindexobservations = unique(sort([td1index,td2index])); % Positions in t for observing error graphs at.
NO = length(tindexobservations); % Number of observation times.

A0hat = A0 - (A1^2+A2^2)/2;

Xrefvalues = zeros(d,NO,runs);

LieA0A1 = A0*A1-A1*A0;
LieA0A2 = A0*A2-A2*A0;
LieA1A2 = A1*A2-A2*A1;

% Initialise the errors, and compile them in vectors, so that we can view
% their means plotted against the step sizes, at the end of this script.
% The first variable refers to the time that the error is observed at. For
% example, ErrorEM(1,.,.) is the error at t = 1, while ErrorEM(2,.,.) is the
% error at t = 2.
ErrorEM = ones(NO,runs,length(hvalues));     % Strong errors compared with X, a reference solution formed by the analytic expression with time step refdt.
ErrorMilSim = ones(NO,runs,length(hvalues));
ErrorMEM = ones(NO,runs,length(hvalues));
ErrorMMSim = ones(NO,runs,length(hvalues));
if useRefinedMilsteinschemes == 1
    ErrorMilRef = ones(NO,runs,length(hvalues));    
    ErrorMMRef = ones(NO,runs,length(hvalues));
end

% Stacks for the values:
EMvalues = zeros(d,NO,runs,length(hvalues));  % Indexing is time, then trial, then step size for the scheme.
MilSimvalues = zeros(d,NO,runs,length(hvalues));
MEMvalues = zeros(d,NO,runs,length(hvalues));
MMSimvalues = zeros(d,NO,runs,length(hvalues));
if useRefinedMilsteinschemes == 1
    MilRefvalues = zeros(d,NO,runs,length(hvalues));
    MMRefvalues = zeros(d,NO,runs,length(hvalues));
end

MSErrorEM = ones(NO,length(hvalues));
MSErrorMilSim = ones(NO,length(hvalues));
MSErrorMEM = ones(NO,length(hvalues));
MSErrorMMSim = ones(NO,length(hvalues));
if useRefinedMilsteinschemes == 1
    MSErrorMilRef = ones(NO,length(hvalues));
    MSErrorMMRef = ones(NO,length(hvalues));
end

%% SIMULATIONS BEGIN HERE:

%% SIMULATIONS BEGIN HERE:

for trial = 1:runs

%% SIMULATION OF WIENER PATH:

    % W1 and W1d1 and W1d2, beginning with W1:
    sqrtrefdt = sqrt(refdt);                    % Value used in the following calculation.
        dW1 = sqrtrefdt*randn(1,length(t));    % Brownian increments.  This is set up so that dW(1,n) = W(1,n+1) - W(1,n),  like DeltaW often is.
        dW1initial = sqrtrefdt*randn(1,1);
        W1initial = 0;
        W1 = zeros(1,length(t));
        for n = 1:length(t)
            if n == 1
                W1(1,n) = dW1initial(1,1);
            else
                W1(1,n) = W1(1,n-1) + dW1(1,n-1);
            end
        end
        % Define W1d1(t) = W1(t-d1) and W1d2(t)=W1(t-d2):
        W1d1 = zeros(1,length(t));
        dW1d1 = zeros(1,length(t));
        for n = 1:length(t)
            if t(n) == t(td1index(1))
                W1d1(1,n) = W1initial;
            elseif t(n) > d1
                W1d1(1,n) = W1(1,n-td1index(1));
            end
            if  td1index(1) <= n && n < length(t)
                %dW1d1(1,n) = W1d1(1,n+1)-W1d1(1,n);
            end
        end
        for n = td1index(1):length(t)-1
            dW1d1(1,n) = W1d1(1,n+1)-W1d1(1,n);
        end
        W1d2 = zeros(1,length(t));
        dW1d2 = zeros(1,length(t));
        for n = 1:length(t)
            if t(n) == t(td2index(1))
                W1d2(1,n) = W1initial;
            elseif t(n) > d2
                W1d2(1,n) = W1(1,n-td2index(1));
            end
            if n < length(t)
                %dW1d2(1,n) = W1d2(1,n+1)-W1d2(1,n);
            end
        end
        for n = td2index(1):length(t)-1
            dW1d2(1,n) = W1d2(1,n+1)-W1d2(1,n);
        end
        % To test these, use the following line in the command window:
        % plot([0,t],[0,W1],'k',[0,t],[0,W1d1],'b',[0,t],[0,W1d2],'r');legend('$W_1$','$W_1^{\tau_1}$','$W_1^{\tau_2}$','Interpreter','latex');if
        % max(W1(1,1:td1index(T-1)) == W1d1(td1index(1)+1:td1index(T))) == 1 &&
        % min(W1(1,1:td1index(T-1)) == W1d1(td1index(1)+1:td1index(T))) == 1;
        % disp('W1 and W1d1 calculated correctly.'); else disp('W1 and W1d1
        % calculated incorrectly.'); end; if max(W1(1,1:td2index(4*T-1)) ==
        % W1d2(td2index(1)+1:end)) == 1 && min(W1(1,1:td2index(4*T-1)) ==
        % W1d2(td2index(1)+1:end)) == 1; disp('W1 and W1d2 calculated
        % correctly.'); else disp('W1 and W1d2 calculated incorrectly.');
        % end; disp('Check values at
        % t=0,refdt,0.25,0.25+refdt,1,1+refdt:'); [{'t';'W1';'W1d1';'W1d2'},{0;0;0;0},{t(1);W1(1);W1d1(1);W1d2(1)},{t(td2index(1));W1(td2index(1));W1d1(td2index(1));W1d2(td2index(1))},{t(td2index(1)+1);W1(td2index(1)+1);W1d1(td2index(1)+1);W1d2(td2index(1)+1)},{t(td1index(1));W1(td1index(1));W1d1(td1index(1));W1d2(td2index(1))},{t(td2index(1)+1);W1(td2index(1)+1);W1d1(td1index(1)+1);W1d2(td2index(1)+1)}]
    % W2, W2d1 and W2d1:
    sqrtrefdt = sqrt(refdt);                    % Value used in the following calculation.
        dW2 = sqrtrefdt*randn(1,length(t));    % Brownian increments.  This is set up so that dW(1,n) = W(1,n+1) - W(1,n),  like DeltaW often is.
        dW2initial = sqrtrefdt*randn(1,1);
        W2initial = 0;
        W2 = zeros(1,length(t));
        for n = 1:length(t)
            if n == 1
                W2(1,n) = dW2initial(1,1);
            else
                W2(1,n) = W2(1,n-1) + dW2(1,n-1);
            end
        end
        % Define W2d1(t) = W2(t-d1) and W2d2(t)=W2(t-d2):
        W2d1 = zeros(1,length(t));
        dW2d1 = zeros(1,length(t));
        for n = 1:length(t)
            if t(n) == t(td1index(1))
                W2d1(1,n) = W2initial;
            elseif t(n) > d1
                W2d1(1,n) = W2(1,n-td1index(1));
            end
            if n < length(t)
                %dW2d1(1,n) = W2d1(1,n+1)-W2d1(1,n);
            end
        end
        for n = td1index(1):length(t)-1
            dW2d1(1,n) = W2d1(1,n+1)-W2d1(1,n);
        end
        W2d2 = zeros(1,length(t));
        dW2d2 = zeros(1,length(t));
        for n = 1:length(t)
            if t(n) == t(td2index(1))
                W2d2(1,n) = W2initial;
            elseif t(n) > d2
                W2d2(1,n) = W2(1,n-td2index(1));
            end
            if n < length(t)
                %dW2d2(1,n) = W2d2(1,n+1)-W2d2(1,n);
            end
        end
        for n = td2index(1):length(t)-1
            dW2d2(1,n) = W2d2(1,n+1)-W2d2(1,n);
        end

%% SIMULATION OF REFERENCE SOLUTION:

    % Construct the reference (analytic) solution (with step size h = refdt):
    Xref = zeros(d,length(t));
    % Initial step:
    Xn = X0; Xd1 = X0; Xd2 = X0;
    a = A0*Xn + B0(Xn,Xd1,Xd2);
    b1 = A1*Xn + B1(Xn,Xd1,Xd2);  b1x = A1+B1x(Xn,Xd1,Xd2);
    b2 = A2*Xn + B2(Xn,Xd1,Xd2);  b2x = A2+B2x(Xn,Xd1,Xd2);
    I11 = (dW1initial(1,1)^2-refdt)/2; I22 = (dW2initial(1,1)^2-refdt)/2;
        % I12ref calculation:
        I12 = dW1initial(1,1)*dW2initial(1,1)/2;
        I21 = dW1initial(1,1)*dW2initial(1,1) - I12;
    % Update these Iijs for these refined ones.
    Lx1b1 = (b1x)*b1;   Lx2b2 = (b2x)*b2;   Lx1b2 = (b2x)*b1;   Lx2b1 = (b1x)*b2;
    Xref(:,1) = Xn + a*refdt + b1*dW1initial(1,1) + b2*dW2initial(1,1) + Lx1b1*I11 + Lx2b1*I21 + Lx1b2*I12 + Lx2b2*I22;
    for n = 1:length(t)-1
        Xn = Xref(:,n);
        if t(n) <= d1 && t(n) <= d2
            Xd1 = X0; Xd1d1 = X0; Xd2 = X0; Xd2d2 = X0;
        elseif t(n) <= d1 && t(n) > d2 && t(n) <= 2*d2
            Xd1 = X0; Xd1d1 = X0; Xd2 = Xref(:,n-td2index(1)); Xd2d2 = X0; 
        elseif t(n) <= d1 && t(n) > 2*d2
            Xd1 = X0; Xd1d1 = X0; Xd2 = Xref(:,n-td2index(1)); Xd2d2 = Xref(:,n-2*td2index(1)); 
        elseif t(n) > d1 && t(n) <= 2*d1 && t(n) <= d2
            Xd1 = Xref(:,n-td1index(1)); Xd1d1 = X0; Xd2 = X0; Xd2d2 = X0; 
        elseif t(n) > d1 && t(n) <= 2*d1 && t(n) > d2 && t(n) <= 2*d2
            Xd1 = Xref(:,n-td1index(1)); Xd1d1 = X0; Xd2 = Xref(:,n-td2index(1)); Xd2d2 = X0; 
        elseif t(n) > d1 && t(n) <= 2*d1 && t(n) > 2*d2
            Xd1 = Xref(:,n-td1index(1)); Xd1d1 = X0; Xd2 = Xref(:,n-td2index(1)); Xd2d2 = Xref(:,n-2*td2index(1)); 
        elseif t(n) > 2*d1 && t(n) <= d2
            Xd1 = Xref(:,n-td1index(1)); Xd1d1 = Xref(:,n-2*td1index(1)); Xd2 = X0; Xd2d2 = X0; 
        elseif t(n) > 2*d1 && t(n) > d2 && t(n) <= 2*d2
            Xd1 = Xref(:,n-td1index(1)); Xd1d1 = Xref(:,n-2*td1index(1)); Xd2 = Xref(:,n-td2index(1)); Xd2d2 = X0; 
        elseif t(n) > 2*d1 && t(n) > 2*d2
            Xd1 = Xref(:,n-td1index(1)); Xd1d1 = Xref(:,n-2*td1index(1)); Xd2 = Xref(:,n-td2index(1)); Xd2d2 = Xref(:,n-2*td2index(1)); 
        end
        if t(n) <= d1+d2        % Do the d1+d2 part.
            Xd1d2 = X0;
        else
            Xd1d2 = Xref(:,n-td1index(1)-td2index(1));
        end
        Xd2d1 = Xd1d2;
        a = A0*Xn + B0(Xn,Xd1,Xd2);
        b1 = A1*Xn + B1(Xn,Xd1,Xd2);  b1x = A1+B1x(Xn,Xd1,Xd2);
        b2 = A2*Xn + B2(Xn,Xd1,Xd2);  b2x = A2+B2x(Xn,Xd1,Xd2);
        I11Ref = (dW1(1,n)^2-refdt)/2; I22Ref = (dW2(1,n)^2-refdt)/2; I12Ref = dW1(1,n)*dW2(1,n)/2; I21Ref = dW1(1,n)*dW2(1,n) - I12Ref;
        Lx1b1 = (b1x)*b1;   Lx2b2 = (b2x)*b2;   Lx1b2 = (b2x)*b1;   Lx2b1 = (b1x)*b2;
        b1xd1 = B1xd1(Xn,Xd1,Xd2); b2xd1 = B2xd1(Xn,Xd1,Xd2); b1d1 = A1*Xd1+ B1(Xd1,Xd1d1,Xd2d1); b2d1 = A2*Xd1+ B2(Xd1,Xd1d1,Xd2d1);
        Lxd_one1b1 = (b1xd1)*b1d1; Lxd_one2b2 = (b2xd1)*b2d1;   Lxd_one1b2 = (b2xd1)*b1d1;   Lxd_one2b1 = (b1xd1)*b2d1;
        if t(n) < d1
            I11d1Ref = 0; I12d1Ref = 0; I21d1Ref = 0; I22d1Ref = 0;
        else
            I11d1Ref = dW1d1(1,n)*dW1(1,n)/2; I12d1Ref = dW1d1(1,n)*dW2(1,n)/2; I21d1Ref = dW1(1,n)*dW2d1(1,n)/2; I22d1Ref = dW2(1,n)*dW2d1(1,n)/2;
        end
        b1xd2 = B1xd2(Xn,Xd1,Xd2);  b2xd2 = B2xd2(Xn,Xd1,Xd2);  b1d2 = A1*Xd2+ B1(Xd2,Xd1d2,Xd2d2); b2d2 = A2*Xd2+ B2(Xd2,Xd1d2,Xd2d2);
        Lxd_two1b1 = (b1xd2)*b1d2; Lxd_two2b2 = (b2xd2)*b2d2;   Lxd_two1b2 = (b2xd2)*b1d2;   Lxd_two2b1 = (b1xd2)*b2d2;  
        if t(n) < d2
            I11d2Ref = 0; I12d2Ref = 0; I21d2Ref = 0; I22d2Ref = 0;
        else
            I11d2Ref = dW1d2(1,n)*dW1(1,n)/2; I12d2Ref = dW1d2(1,n)*dW2(1,n)/2; I21d2Ref = dW1(1,n)*dW2d2(1,n)/2; I22d2Ref = dW2(1,n)*dW2d2(1,n)/2;
        end
        Xref(:,n+1) = Xn + a*refdt + b1*dW1(1,n) + b2*dW2(1,n) + Lx1b1*I11Ref + Lx2b1*I21Ref + Lx1b2*I12Ref + Lx2b2*I22Ref + Lxd_one1b1*I11d1Ref + Lxd_one1b2*I12d1Ref + Lxd_one2b1*I21d1Ref + Lxd_one2b2*I22d1Ref  +  Lxd_two1b1*I11d2Ref + Lxd_two1b2*I12d2Ref + Lxd_two2b1*I21d2Ref + Lxd_two2b2*I22d2Ref;
    end

    for ti = 1:NO
        Xrefvalues(:,ti,trial) = Xref(:,tindexobservations(ti));
    end
    if trial == samplepath
        sampleXref = Xref(samplecomponent,:);
    end

%% SCHEMES:

    for h = 1:length(hvalues)

        % Define some constants that are used throughout the following part:
        step = hvalues(h)/refdt;    % This is the index multiplier, for how many index positions comprise a single step of the simulations.  The idea is that t_n = t(n*step).
        Nd2 = length(t)/step;           % This is the number of steps of length d2 that are made in [0,T].
        Nd1 = Nd2/(d1/d2);           % This is the number of steps of length d1 that are made in [0,T].
        N = length(t)/step;           % This is the number of steps that are made in [0,T].
        Delt = hvalues(h);          % Step size.                 
        A0hatDelt = (A0 - A1^2/2 - A2^2/2)*Delt;
        d1schemetimeindex=find(t(step:step:end)==d1);
        d2schemetimeindex=find(t(step:step:end)==d2);

        % DelW1 and DelW1d1 and DelW1d2:
        DelW1 = zeros(1,N-1);  % Define Delta W1 (see page 340 of Kloeden and Platen).
        DelW1initial = zeros(1,1);
        DelW1d1initial = zeros(1,1);
        DelW1initial(1,1) = W1(1,step);
        for n = 1:N-1
            DelW1(1,n) = W1(1,(n+1)*step) - W1(1,n*step);
        end
        DelW1d1 = zeros(1,N-1);
        for n = 1:N-1
            if t(n*step) == t(td1index(1))
                DelW1d1(1,n) = W1(1,step);
            elseif t(n*step) > t(td1index(1))
                DelW1d1(1,n) = W1(1,(n+1)*step-td1index(1)) - W1(1,n*step-td1index(1));
                %DelW1d1(1,n) = W1(1,(n+1)*step-d1schemetimeindex)-W1(1,n*step-d1schemetimeindex);
            end
        end
        DelW1d2 = zeros(1,N-1);
        for n = 1:N-1
            if t(n*step) == t(td2index(1))
                DelW1d2(1,n) = W1(1,step);
            elseif t(n*step) > t(td2index(1))
                DelW1d2(1,n) = W1(1,(n+1)*step-td2index(1)) - W1(1,n*step-td2index(1));
                %DelW1d2(1,n) = W1(1,(n+1)*step-d2schemetimeindex)-W1(1,n*step-d2schemetimeindex);
            end
        end
        % To test this in the command window:
        % hold on; plot([0,t],[0,W1],'k'); plot([0,t],[0,W1d1],'r'); plot([0,t],[0,W1d2],'b'); plot([t(step:step:end-step)],[0,cumsum(DelW1d2(1:end-1))],'r:'); legend('$W_1$','$W_1^{\tau_1}$','$W_1^{\tau_2}$','$W_1^{\tau_2}$ from $\Delta W_1^{\tau_2}$','Interpreter','latex')

        % DelW2 and DelW2d1 and DelW2d2:
        DelW2 = zeros(1,N-1);  % Define Delta W2 (see page 340 of Kloeden and Platen).
        DelW2initial = zeros(1,1);
        DelW2d1initial = zeros(1,1);
        DelW2initial(1,1) = W2(1,step);
        for n = 1:N-1
            DelW2(1,n) = W2(1,(n+1)*step) - W2(1,n*step);
        end
        DelW2d1 = zeros(1,N-1);
        for n = 1:N-1
            if t(n*step) == t(td1index(1))
                DelW2d1(1,n) = W2(1,step);
            elseif t(n*step) > t(td1index(1))
                DelW2d1(1,n) = W2(1,(n+1)*step-td1index(1)) - W2(1,n*step-td1index(1));
                %DelW2d1(1,n) = W2(1,(n+1)*step-d1schemetimeindex)-W2(1,n*step-d1schemetimeindex);
            end
        end
        DelW2d2 = zeros(1,N-1);
        for n = 1:N-1
            if t(n*step) == t(td2index(1))
                DelW2d2(1,n) = W2(1,step);
            elseif t(n*step) > t(td2index(1))
                DelW2d2(1,n) = W2(1,(n+1)*step-td2index(1)) - W2(1,n*step-td2index(1));
                %DelW2d2(1,n) = W2(1,(n+1)*step-d2schemetimeindex)-W2(1,n*step-d2schemetimeindex);
            end
        end

        % Initialise the schemes:
        EM = zeros(d,N);       % Euler--Maruyama 
        MilSim = zeros(d,N);   % Milstein (Simple)
        MEM = zeros(d,N);      % Magnus--EM
        MMSim = zeros(d,N);    % Magnus--Mistein (Simple)
        if useRefinedMilsteinschemes == 1
            MilRef = zeros(d,N);   % Milstein (Refined)
            MMRef = zeros(d,N);    % Magnus--Milstein (Refined)
        end

        % The first step, from t=0 to t=h, using initial increments, such
        % as DelW1initial.  Effectively, this is n = 0:

    % INITIAL STEP:

        % EM:
        Yn = X0; Yd1 = X0; Yd2 = X0;
        a = A0*Yn + B0(Yn,Yd1,Yd2);
        b1 = A1*Yn + B1(Yn,Yd1,Yd2);
        b2 = A2*Yn + B2(Yn,Yd1,Yd2);
        EM(:,1) = X0 + a*Delt + b1*DelW1initial(1,1) + b2*DelW2initial(1,1);

        % MilSim:
        Yn = X0; Yd1 = X0; Yd2 = X0;
        a = A0*Yn + B0(Yn,Yd1,Yd2);
        b1 = A1*Yn + B1(Yn,Yd1,Yd2); b1x = A1+B1x(Yn,Yd1,Yd2);
        b2 = A2*Yn + B2(Yn,Yd1,Yd2); b2x = A2+B2x(Yn,Yd1,Yd2);
        I11Sim = (DelW1initial(1,1)^2-Delt)/2; I22Sim = (DelW2initial(1,1)^2-Delt)/2; I12Sim = DelW1initial(1,1)*DelW2initial(1,1)/2; I21Sim = DelW1initial(1,1)*DelW2initial(1,1)/2;
        Lx1b1 = (b1x)*b1;   Lx2b2 = (b2x)*b2;   Lx1b2 = (b2x)*b1;   Lx2b1 = (b1x)*b2;
        MilSim(:,1) = Yn + a*Delt + b1*DelW1initial(1,1) + b2*DelW2initial(1,1) + Lx1b1*I11Sim + Lx2b1*I21Sim + Lx1b2*I12Sim + Lx2b2*I22Sim;
        
        % MEM:
        Yn = X0; Yd1 = X0; Yd2 = X0;
        a = B0(Yn,Yd1,Yd2);
        b1 = B1(Yn,Yd1,Yd2);
        b2 = B2(Yn,Yd1,Yd2);
        expOmega1 = expm( A0hatDelt + A1*DelW1initial(1,1) + A2*DelW2initial(1,1) );
        MEM(:,1) = expOmega1*(Yn + (a - A1*b1 - A2*b2)*Delt + b1*DelW1initial(1,1) + b2*DelW2initial(1,1) );
        
        % MMSim:
        Yn = X0; Yd1 = X0; Yd2 = X0;
        a = B0(Yn,Yd1,Yd2);
        b1 = B1(Yn,Yd1,Yd2); b1x = B1x(Yn,Yd1,Yd2);
        b2 = B2(Yn,Yd1,Yd2); b2x = B2x(Yn,Yd1,Yd2);
        I10Sim = DelW1initial(1,1)*Delt/2; I01Sim = DelW1initial(1,1)*Delt - I10Sim; I20Sim = DelW2initial(1,1)*Delt/2; I02Sim = DelW2initial(1,1)*Delt - I20Sim;
        expOmega2 = expm( A0hatDelt + A1*DelW1initial(1,1) + A2*DelW2initial(1,1) + 1/2*( LieA0A1*(I10Sim-I01Sim)+LieA0A2*(I20Sim-I02Sim)+LieA1A2*(I21Sim-I12Sim) ) );
        Hx1b1 = (b1x)*(A1*Yn+b1)-A1*b1; % This 0 is actually the Jacobian, d(b1)/dx.   j = 1, l = 1
        Hx1b2 = (b1x)*(A2*Yn+b2)-A2*b1; %  j = 1, l = 2
        Hx2b1 = (b2x)*(A1*Yn+b1)-A1*b2; %  j = 2, l = 1
        Hx2b2 = (b2x)*(A2*Yn+b2)-A2*b2; %  j = 2, l = 1
        MMSim(:,1) = expOmega2*(Yn + (a - A1*b1 - A2*b2)*Delt + b1*DelW1initial(1,1) + b2*DelW2initial(1,1) + Hx1b1*I11Sim + Hx1b2*I21Sim + Hx2b1*I12Sim + Hx2b2*I22Sim );
        
        if useRefinedMilsteinschemes == 1
            
            % MilRef:
            Yn = X0; Yd1 = X0; Yd2 = X0;
            a = A0*Yn + B0(Yn,Yd1,Yd2);
            b1 = A1*Yn + B1(Yn,Yd1,Yd2); b1x = A1+B1x(Yn,Yd1,Yd2);
            b2 = A2*Yn + B2(Yn,Yd1,Yd2); b2x = A2+B2x(Yn,Yd1,Yd2);
            I11Ref = (DelW1initial(1,1)^2-Delt)/2; I22Ref = (DelW2initial(1,1)^2-Delt)/2;
                % I12ref calculation:
                I12Ref = dW1initial(1,1)*dW2initial(1,1)/2;
                for j=1:step-1
                    I12Ref = I12Ref + dW1(1,j)*dW2(1,j)/2 + dW2(1,j)*(W1(1,j)-W1initial(1,1));
                end
                I21Ref = DelW1initial(1,1)*DelW2initial(1,1) - I12Ref;
            Lx1b1 = (b1x)*b1;   Lx2b2 = (b2x)*b2;   Lx1b2 = (b2x)*b1;   Lx2b1 = (b1x)*b2;
            MilRef(:,1) = Yn + a*Delt + b1*DelW1initial(1,1) + b2*DelW2initial(1,1) + Lx1b1*I11Ref + Lx2b1*I21Ref + Lx1b2*I12Ref + Lx2b2*I22Ref;

            % MMRef:
            Yn = X0; Yd1 = X0; Yd2 = X0;
            a = B0(Yn,Yd1,Yd2);
            b1 = B1(Yn,Yd1,Yd2); b1x = B1x(Yn,Yd1,Yd2);
            b2 = B2(Yn,Yd1,Yd2); b2x = B2x(Yn,Yd1,Yd2);
            I10Ref = dW1initial(1,1)*refdt/2;
                for j=1:step-1
                    I10Ref = I10Ref + dW1(1,j)*refdt/2 + refdt*(W1(1,j)-W1initial(1,1));
                end
            I01Ref = DelW1initial(1,1)*Delt - I10Ref;
            I20Ref = dW2initial(1,1)*refdt/2;
                for j=1:step-1
                    I20Ref = I20Ref + dW2(1,j)*refdt/2 + refdt*(W2(1,j)-W2initial(1,1));
                end
            I02Ref = DelW2initial(1,1)*Delt - I20Ref;
            expOmega2 = expm( A0hatDelt + A1*DelW1initial(1,1) + A2*DelW2initial(1,1) + 1/2*( LieA0A1*(I10Ref-I01Ref)+LieA0A2*(I20Ref-I02Ref)+LieA1A2*(I21Ref-I12Ref) ) );
            Hx1b1 = (b1x)*(A1*Yn+b1)-A1*b1; % This 0 is actually the Jacobian, d(b1)/dx.   j = 1, l = 1
            Hx1b2 = (b1x)*(A2*Yn+b2)-A2*b1; %  j = 1, l = 2
            Hx2b1 = (b2x)*(A1*Yn+b1)-A1*b2; %  j = 2, l = 1
            Hx2b2 = (b2x)*(A2*Yn+b2)-A2*b2; %  j = 2, l = 1
            MMRef(:,1) = expOmega2*(Yn + (a - A1*b1 - A2*b2)*Delt + b1*DelW1initial(1,1) + b2*DelW2initial(1,1) + Hx1b1*I11Ref + Hx1b2*I21Ref + Hx2b1*I12Ref + Hx2b2*I22Ref );

        end

    % ITERATIVE STEPS:

        for n = 1:N-1
        
            % EM:
            Yn = EM(:,n);
            if t(n*step) <= d1 && t(n*step) <= d2
                Yd1 = X0; Yd1d1 = X0; Yd2 = X0; Yd2d2 = X0;
            elseif t(n*step) <= d1 && t(n*step) > d2 && t(n*step) <= 2*d2
                Yd1 = X0; Yd1d1 = X0; Yd2 = EM(:,n-d2schemetimeindex); Yd2d2 = X0;
            elseif t(n*step) <= d1 && t(n*step) > 2*d2
                Yd1 = X0; Yd1d1 = X0; Yd2 = EM(:,n-d2schemetimeindex); Yd2d2 = EM(:,n-2*d2schemetimeindex);
            elseif t(n*step) > d1 && t(n*step) <= 2*d1 && t(n*step) <= d2
                Yd1 = EM(:,n-d1schemetimeindex); Yd1d1 = X0; Yd2 = X0; Yd2d2 = X0;
            elseif t(n*step) > d1 && t(n*step) <= 2*d1 && t(n*step) > d2 && t(n*step) <= 2*d2
                Yd1 = EM(:,n-d1schemetimeindex); Yd1d1 = X0; Yd2 = EM(:,n-d2schemetimeindex); Yd2d2 = X0;
            elseif t(n*step) > d1 && t(n*step) <= 2*d1 && t(n*step) > 2*d2
                Yd1 = EM(:,n-d1schemetimeindex); Yd1d1 = X0; Yd2 = EM(:,n-d2schemetimeindex); Yd2d2 = EM(:,n-2*d2schemetimeindex);
            elseif t(n*step) > 2*d1 && t(n*step) <= d2
                Yd1 = EM(:,n-d1schemetimeindex); Yd1d1 = EM(:,n-2*d1schemetimeindex); Yd2 = X0; Yd2d2 = X0;
            elseif t(n*step) > 2*d1 && t(n*step) > d2 && t(n*step) <= 2*d2
                Yd1 = EM(:,n-d1schemetimeindex); Yd1d1 = EM(:,n-2*d1schemetimeindex); Yd2 = EM(:,n-d2schemetimeindex); Yd2d2 = X0;
            elseif t(n*step) > 2*d1 && t(n*step) > 2*d2
                Yd1 = EM(:,n-d1schemetimeindex); Yd1d1 = EM(:,n-2*d1schemetimeindex); Yd2 = EM(:,n-d2schemetimeindex); Yd2d2 = EM(:,n-2*d2schemetimeindex);
            end
            if t(n*step) <= d1+d2
                Yd1d2 = X0; Yd2d1 = X0;
            elseif t(n*step) > d1+d2
                Yd1d2 = EM(:,n-d1schemetimeindex-d2schemetimeindex); Yd2d1 = Yd1d2;
            end
            a = A0*Yn + B0(Yn,Yd1,Yd2);
            b1 = A1*Yn + B1(Yn,Yd1,Yd2);
            b2 = A2*Yn + B2(Yn,Yd1,Yd2);
            EM(:,n+1) = Yn + a*Delt + b1*DelW1(1,n) + b2*DelW2(1,n);

            % MilSim:
            Yn = MilSim(:,n);
            if t(n*step) <= d1 && t(n*step) <= d2
                Yd1 = X0; Yd1d1 = X0; Yd2 = X0; Yd2d2 = X0;
            elseif t(n*step) <= d1 && t(n*step) > d2 && t(n*step) <= 2*d2
                Yd1 = X0; Yd1d1 = X0; Yd2 = MilSim(:,n-d2schemetimeindex); Yd2d2 = X0;
            elseif t(n*step) <= d1 && t(n*step) > 2*d2
                Yd1 = X0; Yd1d1 = X0; Yd2 = MilSim(:,n-d2schemetimeindex); Yd2d2 = MilSim(:,n-2*d2schemetimeindex);
            elseif t(n*step) > d1 && t(n*step) <= 2*d1 && t(n*step) <= d2
                Yd1 = MilSim(:,n-d1schemetimeindex); Yd1d1 = X0; Yd2 = X0; Yd2d2 = X0;
            elseif t(n*step) > d1 && t(n*step) <= 2*d1 && t(n*step) > d2 && t(n*step) <= 2*d2
                Yd1 = MilSim(:,n-d1schemetimeindex); Yd1d1 = X0; Yd2 = MilSim(:,n-d2schemetimeindex); Yd2d2 = X0;
            elseif t(n*step) > d1 && t(n*step) <= 2*d1 && t(n*step) > 2*d2
                Yd1 = MilSim(:,n-d1schemetimeindex); Yd1d1 = X0; Yd2 = MilSim(:,n-d2schemetimeindex); Yd2d2 = MilSim(:,n-2*d2schemetimeindex);
            elseif t(n*step) > 2*d1 && t(n*step) <= d2
                Yd1 = MilSim(:,n-d1schemetimeindex); Yd1d1 = MilSim(:,n-2*d1schemetimeindex); Yd2 = X0; Yd2d2 = X0;
            elseif t(n*step) > 2*d1 && t(n*step) > d2 && t(n*step) <= 2*d2
                Yd1 = MilSim(:,n-d1schemetimeindex); Yd1d1 = MilSim(:,n-2*d1schemetimeindex); Yd2 = MilSim(:,n-d2schemetimeindex); Yd2d2 = X0;
            elseif t(n*step) > 2*d1 && t(n*step) > 2*d2
                Yd1 = MilSim(:,n-d1schemetimeindex); Yd1d1 = MilSim(:,n-2*d1schemetimeindex); Yd2 = MilSim(:,n-d2schemetimeindex); Yd2d2 = MilSim(:,n-2*d2schemetimeindex);
            end
            if t(n*step) <= d1+d2
                Yd1d2 = X0; Yd2d1 = X0;
            elseif t(n*step) > d1+d2
                Yd1d2 = MilSim(:,n-d1schemetimeindex-d2schemetimeindex); Yd2d1 = Yd1d2;
            end
            %a,b1,b2,...
            a = A0*Yn + B0(Yn,Yd1,Yd2);
            b1 = A1*Yn + B1(Yn,Yd1,Yd2); b1x = A1+B1x(Yn,Yd1,Yd2);
            b2 = A2*Yn + B2(Yn,Yd1,Yd2); b2x = A2+B2x(Yn,Yd1,Yd2);
            I11Sim = (DelW1(1,n)^2-Delt)/2; I22Sim = (DelW2(1,n)^2-Delt)/2; I12Sim = DelW1(1,n)*DelW2(1,n)/2; I21Sim = DelW1(1,n)*DelW2(1,n)/2;
            Lx1b1 = (b1x)*b1;   Lx2b2 = (b2x)*b2;   Lx1b2 = (b2x)*b1;   Lx2b1 = (b1x)*b2;
            b1xd1 = B1xd1(Yn,Yd1,Yd2); b2xd1 = B2xd1(Yn,Yd1,Yd2); b1d1 = A1*Yd1 + B1(Yd1,Yd1d1,Yd2d1); b2d1 = A2*Yd1 + B2(Yd1,Yd1d1,Yd2d1);
            Lxd_one1b1 = (b1xd1)*b1d1; Lxd_one2b2 = (b2xd1)*b2d1;   Lxd_one1b2 = (b2xd1)*b1d1;   Lxd_one2b1 = (b1xd1)*b2d1;
            if t(n*step) < d1
                I11d1Sim = 0; I22d1Sim = 0; I12d1Sim = 0; I21d1Sim = 0;
            else
                I11d1Sim = DelW1d1(1,n)*DelW1(1,n)/2; I22d1Sim = DelW2d1(1,n)*DelW2(1,n)/2; I12d1Sim = DelW1d1(1,n)*DelW2(1,n)/2; I21d1Sim = DelW2d1(1,n)*DelW1(1,n)/2;
            end
            b1xd2 = B1xd2(Yn,Yd1,Yd2);  b2xd2 = B2xd2(Yn,Yd1,Yd2);  b1d2 = A1*Yd2 + B1(Yd2,Yd1d2,Yd2d2); b2d2 = A2*Yd2 + B2(Yd2,Yd1d2,Yd2d2);
            Lxd_two1b1 = (b1xd2)*b1d2; Lxd_two2b2 = (b2xd2)*b2d2;   Lxd_two1b2 = (b2xd2)*b1d2;   Lxd_two2b1 = (b1xd2)*b2d2;  
            if t(n*step) < d2 
                I11d2Sim = 0; I22d2Sim = 0; I12d2Sim = 0; I21d2Sim = 0;
            else
                I11d2Sim = DelW1d2(1,n)*DelW1(1,n)/2; I22d2Sim = DelW2d2(1,n)*DelW2(1,n)/2; I12d2Sim = DelW1d2(1,n)*DelW2(1,n)/2; I21d2Sim = DelW2d2(1,n)*DelW1(1,n)/2;
            end
            MilSim(:,n+1) = Yn + a*Delt + b1*DelW1(1,n) + b2*DelW2(1,n) + Lx1b1*I11Sim + Lx1b2*I12Sim + Lx2b1*I21Sim + Lx2b2*I22Sim + Lxd_one1b1*I11d1Sim + Lxd_one1b2*I12d1Sim + Lxd_one2b1*I21d1Sim + Lxd_one2b2*I22d1Sim  +  Lxd_two1b1*I11d2Sim + Lxd_two1b2*I12d2Sim + Lxd_two2b1*I21d2Sim + Lxd_two2b2*I22d2Sim;
          
            % MEM:
            Yn = MEM(:,n);
            if t(n*step) <= d1 && t(n*step) <= d2
                Yd1 = X0; Yd1d1 = X0; Yd2 = X0; Yd2d2 = X0;
            elseif t(n*step) <= d1 && t(n*step) > d2 && t(n*step) <= 2*d2
                Yd1 = X0; Yd1d1 = X0; Yd2 = MEM(:,n-d2schemetimeindex); Yd2d2 = X0;
            elseif t(n*step) <= d1 && t(n*step) > 2*d2
                Yd1 = X0; Yd1d1 = X0; Yd2 = MEM(:,n-d2schemetimeindex); Yd2d2 = MEM(:,n-2*d2schemetimeindex);
            elseif t(n*step) > d1 && t(n*step) <= 2*d1 && t(n*step) <= d2
                Yd1 = MEM(:,n-d1schemetimeindex); Yd1d1 = X0; Yd2 = X0; Yd2d2 = X0;
            elseif t(n*step) > d1 && t(n*step) <= 2*d1 && t(n*step) > d2 && t(n*step) <= 2*d2
                Yd1 = MEM(:,n-d1schemetimeindex); Yd1d1 = X0; Yd2 = MEM(:,n-d2schemetimeindex); Yd2d2 = X0;
            elseif t(n*step) > d1 && t(n*step) <= 2*d1 && t(n*step) > 2*d2
                Yd1 = MEM(:,n-d1schemetimeindex); Yd1d1 = X0; Yd2 = MEM(:,n-d2schemetimeindex); Yd2d2 = MEM(:,n-2*d2schemetimeindex);
            elseif t(n*step) > 2*d1 && t(n*step) <= d2
                Yd1 = MEM(:,n-d1schemetimeindex); Yd1d1 = MEM(:,n-2*d1schemetimeindex); Yd2 = X0; Yd2d2 = X0;
            elseif t(n*step) > 2*d1 && t(n*step) > d2 && t(n*step) <= 2*d2
                Yd1 = MEM(:,n-d1schemetimeindex); Yd1d1 = MEM(:,n-2*d1schemetimeindex); Yd2 = MEM(:,n-d2schemetimeindex); Yd2d2 = X0;
            elseif t(n*step) > 2*d1 && t(n*step) > 2*d2
                Yd1 = MEM(:,n-d1schemetimeindex); Yd1d1 = MEM(:,n-2*d1schemetimeindex); Yd2 = MEM(:,n-d2schemetimeindex); Yd2d2 = MEM(:,n-2*d2schemetimeindex);
            end
            if t(n*step) <= d1+d2
                Yd1d2 = X0; Yd2d1 = X0;
            elseif t(n*step) > d1+d2
                Yd1d2 = MEM(:,n-d1schemetimeindex-d2schemetimeindex); Yd2d1 = Yd1d2;
            end
            a = B0(Yn,Yd1,Yd2);
            b1 = B1(Yn,Yd1,Yd2);
            b2 = B2(Yn,Yd1,Yd2);
            expOmega1 = expm( A0hatDelt + A1*DelW1(1,n) + A2*DelW2(1,n) );
            MEM(:,n+1) = expOmega1*(Yn + (a - A1*b1 - A2*b2)*Delt + b1*DelW1(1,n) + b2*DelW2(1,n) );
            
            % MMSim:
            Yn = MMSim(:,n);
            if t(n*step) <= d1 && t(n*step) <= d2
                Yd1 = X0; Yd1d1 = X0; Yd2 = X0; Yd2d2 = X0;
            elseif t(n*step) <= d1 && t(n*step) > d2 && t(n*step) <= 2*d2
                Yd1 = X0; Yd1d1 = X0; Yd2 = MMSim(:,n-d2schemetimeindex); Yd2d2 = X0;
            elseif t(n*step) <= d1 && t(n*step) > 2*d2
                Yd1 = X0; Yd1d1 = X0; Yd2 = MMSim(:,n-d2schemetimeindex); Yd2d2 = MMSim(:,n-2*d2schemetimeindex);
            elseif t(n*step) > d1 && t(n*step) <= 2*d1 && t(n*step) <= d2
                Yd1 = MMSim(:,n-d1schemetimeindex); Yd1d1 = X0; Yd2 = X0; Yd2d2 = X0;
            elseif t(n*step) > d1 && t(n*step) <= 2*d1 && t(n*step) > d2 && t(n*step) <= 2*d2
                Yd1 = MMSim(:,n-d1schemetimeindex); Yd1d1 = X0; Yd2 = MMSim(:,n-d2schemetimeindex); Yd2d2 = X0;
            elseif t(n*step) > d1 && t(n*step) <= 2*d1 && t(n*step) > 2*d2
                Yd1 = MMSim(:,n-d1schemetimeindex); Yd1d1 = X0; Yd2 = MMSim(:,n-d2schemetimeindex); Yd2d2 = MMSim(:,n-2*d2schemetimeindex);
            elseif t(n*step) > 2*d1 && t(n*step) <= d2
                Yd1 = MMSim(:,n-d1schemetimeindex); Yd1d1 = MMSim(:,n-2*d1schemetimeindex); Yd2 = X0; Yd2d2 = X0;
            elseif t(n*step) > 2*d1 && t(n*step) > d2 && t(n*step) <= 2*d2
                Yd1 = MMSim(:,n-d1schemetimeindex); Yd1d1 = MMSim(:,n-2*d1schemetimeindex); Yd2 = MMSim(:,n-d2schemetimeindex); Yd2d2 = X0;
            elseif t(n*step) > 2*d1 && t(n*step) > 2*d2
                Yd1 = MMSim(:,n-d1schemetimeindex); Yd1d1 = MMSim(:,n-2*d1schemetimeindex); Yd2 = MMSim(:,n-d2schemetimeindex); Yd2d2 = MMSim(:,n-2*d2schemetimeindex);
            end
            if t(n*step) <= d1+d2
                Yd1d2 = X0; Yd2d1 = X0;
            elseif t(n*step) > d1+d2
                Yd1d2 = MMSim(:,n-d1schemetimeindex-d2schemetimeindex); Yd2d1 = Yd1d2;
            end
            %a,b1,b2,...
            a = B0(Yn,Yd1,Yd2);
            b1 = B1(Yn,Yd1,Yd2); b1x = B1x(Yn,Yd1,Yd2);
            b2 = B2(Yn,Yd1,Yd2); b2x = B2x(Yn,Yd1,Yd2);
            I10Sim = DelW1(1,n)*Delt/2; I01Sim = DelW1(1,n)*Delt - I10Sim; I20Sim = DelW2(1,n)*Delt/2; I02Sim = DelW2(1,n)*Delt - I20Sim;
            expOmega2 = expm( A0hatDelt + A1*DelW1(1,n) + A2*DelW2(1,n) + 1/2*( LieA0A1*(I10Sim-I01Sim)+LieA0A2*(I20Sim-I02Sim)+LieA1A2*(I21Sim-I12Sim) ) );
            Hx1b1 = (b1x)*(A1*Yn+b1)-A1*b1; % This 0 is actually the Jacobian, d(b1)/dx.   j = 1, l = 1
            Hx1b2 = (b1x)*(A2*Yn+b2)-A2*b1; %  j = 1, l = 2
            Hx2b1 = (b2x)*(A1*Yn+b1)-A1*b2; %  j = 2, l = 1
            Hx2b2 = (b2x)*(A2*Yn+b2)-A2*b2; %  j = 2, l = 1
            b1d1 = B1(Yd1,Yd1d1,Yd2d1); b2d1 = B2(Yd1,Yd1d1,Yd2d1);
            Hxd_one1b1 = B1xd1(Yn,Yd1,Yd2)*(A1*Yd1+b1d1);  % j = 1; l = 1;
            Hxd_one1b2 = B1xd1(Yn,Yd1,Yd2)*(A2*Yd1+b2d1);  % j = 1; l = 2;
            Hxd_one2b1 = B2xd1(Yn,Yd1,Yd2)*(A1*Yd1+b1d1);  % j = 2; l = 1;
            Hxd_one2b2 = B2xd1(Yn,Yd1,Yd2)*(A2*Yd1+b2d1);  % j = 2; l = 2;
            b1d2 = B1(Yd2,Yd1d2,Yd2d2); b2d2 = B2(Yd2,Yd1d2,Yd2d2);
            Hxd_two1b1 = B1xd2(Yn,Yd1,Yd2)*(A1*Yd2+b1d2);  % j = 1; l = 1;        
            Hxd_two1b2 = B1xd2(Yn,Yd1,Yd2)*(A2*Yd2+b2d2);  % j = 1; l = 2;       
            Hxd_two2b1 = B2xd2(Yn,Yd1,Yd2)*(A1*Yd2+b1d2);  % j = 2; l = 1;
            Hxd_two2b2 = B2xd2(Yn,Yd1,Yd2)*(A2*Yd2+b2d2);  % j = 2; l = 2;
            MMSim(:,n+1) = expOmega2*(Yn + (a - A1*b1 - A2*b2)*Delt + b1*DelW1(1,n) + b2*DelW2(1,n) + Hx1b1*I11Sim + Hx1b2*I21Sim + Hx2b1*I12Sim + Hx2b2*I22Sim + Hxd_one1b1*I11d1Sim + Hxd_one2b1*I12d1Sim + Hxd_one1b2*I21d1Sim + Hxd_one2b2*I22d1Sim  +  Hxd_two1b1*I11d2Sim + Hxd_two2b1*I12d2Sim + Hxd_two1b2*I21d2Sim + Hxd_two2b2*I22d2Sim );
            
            if useRefinedMilsteinschemes == 1

                % MilRef:
                Yn = MilRef(:,n);
                if t(n*step) <= d1 && t(n*step) <= d2
                    Yd1 = X0; Yd1d1 = X0; Yd2 = X0; Yd2d2 = X0;
                elseif t(n*step) <= d1 && t(n*step) > d2 && t(n*step) <= 2*d2
                    Yd1 = X0; Yd1d1 = X0; Yd2 = MilRef(:,n-d2schemetimeindex); Yd2d2 = X0;
                elseif t(n*step) <= d1 && t(n*step) > 2*d2
                    Yd1 = X0; Yd1d1 = X0; Yd2 = MilRef(:,n-d2schemetimeindex); Yd2d2 = MilRef(:,n-2*d2schemetimeindex);
                elseif t(n*step) > d1 && t(n*step) <= 2*d1 && t(n*step) <= d2
                    Yd1 = MilRef(:,n-d1schemetimeindex); Yd1d1 = X0; Yd2 = X0; Yd2d2 = X0;
                elseif t(n*step) > d1 && t(n*step) <= 2*d1 && t(n*step) > d2 && t(n*step) <= 2*d2
                    Yd1 = MilRef(:,n-d1schemetimeindex); Yd1d1 = X0; Yd2 = MilRef(:,n-d2schemetimeindex); Yd2d2 = X0;
                elseif t(n*step) > d1 && t(n*step) <= 2*d1 && t(n*step) > 2*d2
                    Yd1 = MilRef(:,n-d1schemetimeindex); Yd1d1 = X0; Yd2 = MilRef(:,n-d2schemetimeindex); Yd2d2 = MilRef(:,n-2*d2schemetimeindex);
                elseif t(n*step) > 2*d1 && t(n*step) <= d2
                    Yd1 = MilRef(:,n-d1schemetimeindex); Yd1d1 = MilRef(:,n-2*d1schemetimeindex); Yd2 = X0; Yd2d2 = X0;
                elseif t(n*step) > 2*d1 && t(n*step) > d2 && t(n*step) <= 2*d2
                    Yd1 = MilRef(:,n-d1schemetimeindex); Yd1d1 = MilRef(:,n-2*d1schemetimeindex); Yd2 = MilRef(:,n-d2schemetimeindex); Yd2d2 = X0;
                elseif t(n*step) > 2*d1 && t(n*step) > 2*d2
                    Yd1 = MilRef(:,n-d1schemetimeindex); Yd1d1 = MilRef(:,n-2*d1schemetimeindex); Yd2 = MilRef(:,n-d2schemetimeindex); Yd2d2 = MilRef(:,n-2*d2schemetimeindex);
                end
                if t(n*step) <= d1+d2
                    Yd1d2 = X0; Yd2d1 = X0;
                elseif t(n*step) > d1+d2
                    Yd1d2 = MilRef(:,n-d1schemetimeindex-d2schemetimeindex); Yd2d1 = Yd1d2;
                end
                %a,b1,b2,...
                a = A0*Yn + B0(Yn,Yd1,Yd2);
                b1 = A1*Yn + B1(Yn,Yd1,Yd2); b1x = A1+B1x(Yn,Yd1,Yd2);
                b2 = A2*Yn + B2(Yn,Yd1,Yd2); b2x = A2+B2x(Yn,Yd1,Yd2);
                I11Ref = (DelW1(1,n)^2-Delt)/2; I22Ref = (DelW2(1,n)^2-Delt)/2;
                    % I12ref calculation:
                    I12Ref = dW1(1,n*step)*dW2(1,n*step)/2;
                    for j=1:step-1
                        I12Ref = I12Ref + dW1(1,n*step+j)*dW2(1,n*step+j)/2 + dW2(1,n*step+j)*(W1(1,n*step+j)-W1(1,n*step));
                    end
                    I21Ref = DelW1(1,n)*DelW2(1,n) - I12Ref;
                Lx1b1 = (b1x)*b1;   Lx2b2 = (b2x)*b2;   Lx1b2 = (b2x)*b1;   Lx2b1 = (b1x)*b2;
                b1xd1 = B1xd1(Yn,Yd1,Yd2); b2xd1 = B2xd1(Yn,Yd1,Yd2); b1d1 = A1*Yd1 + B1(Yd1,Yd1d1,Yd2d1); b2d1 = A2*Yd1 + B2(Yd1,Yd1d1,Yd2d1);
                Lxd_one1b1 = (b1xd1)*b1d1; Lxd_one2b2 = (b2xd1)*b2d1;   Lxd_one1b2 = (b2xd1)*b1d1;   Lxd_one2b1 = (b1xd1)*b2d1;
                % Iijd1Ref calculations:
                if t(n*step) < d1
                    I11d1Ref = 0; I12d1Ref = 0; I21d1Ref = 0; I22d1Ref = 0;
                else
                    % I11d1Ref:
                        I11d1Ref = dW1d1(1,n*step)*dW1(1,n*step)/2;
                        for j=1:step-1
                            I11d1Ref = I11d1Ref + dW1d1(1,n*step+j)*dW1(1,n*step+j)/2 + dW1(1,n*step+j)*(W1d1(1,n*step+j)-W1d1(1,n*step));
                        end
                    % I12d1Ref:
                        I12d1Ref = dW1d1(1,n*step)*dW2(1,n*step)/2;
                        for j=1:step-1
                            I12d1Ref = I12d1Ref + dW1d1(1,n*step+j)*dW2(1,n*step+j)/2 + dW2(1,n*step+j)*(W1d1(1,n*step+j)-W1d1(1,n*step));
                        end
                    % I21d1Ref:
                        I21d1Ref = dW1(1,n*step)*dW2d1(1,n*step)/2;
                        for j=1:step-1
                            I21d1Ref = I21d1Ref + dW1(1,n*step+j)*dW2d1(1,n*step+j)/2 + dW2d1(1,n*step+j)*(W1(1,n*step+j)-W1(1,n*step));
                        end
                    % I22d1Ref:
                        I22d1Ref = dW2(1,n*step)*dW2d1(1,n*step)/2;
                        for j=1:step-1
                            I22d1Ref = I22d1Ref + dW2(1,n*step+j)*dW2d1(1,n*step+j)/2 + dW2d1(1,n*step+j)*(W2(1,n*step+j)-W2(1,n*step));
                        end
                    I21d1Ref = DelW1(1,n)*DelW2d1(1,n)-I21d1Ref;
                    I22d1Ref = DelW2(1,n)*DelW2d1(1,n)-I22d1Ref;
                end
                b1xd2 = B1xd2(Yn,Yd1,Yd2);  b2xd2 = B2xd2(Yn,Yd1,Yd2);  b1d2 = A1*Yd2 + B1(Yd2,Yd1d2,Yd2d2); b2d2 = A2*Yd2 + B2(Yd2,Yd1d2,Yd2d2);
                Lxd_two1b1 = (b1xd2)*b1d2; Lxd_two2b2 = (b2xd2)*b2d2;   Lxd_two1b2 = (b2xd2)*b1d2;   Lxd_two2b1 = (b1xd2)*b2d2;  
                % Iijd2Ref calculations: 
                if t(n*step) < d2
                    I11d2Ref = 0; I12d2Ref = 0; I21d2Ref = 0; I22d2Ref = 0;
                else
                    % I11d2Ref:
                        I11d2Ref = dW1d2(1,n*step)*dW1(1,n*step)/2;
                        for j=1:step-1
                            I11d2Ref = I11d2Ref + dW1d2(1,n*step+j)*dW1(1,n*step+j)/2 + dW1(1,n*step+j)*(W1d2(1,n*step+j)-W1d2(1,n*step));
                        end
                    % I12d2Ref:
                        I12d2Ref = dW1d2(1,n*step)*dW2(1,n*step)/2;
                        for j=1:step-1
                            I12d2Ref = I12d2Ref + dW1d2(1,n*step+j)*dW2(1,n*step+j)/2 + dW2(1,n*step+j)*(W1d2(1,n*step+j)-W1d2(1,n*step));
                        end
                    % I21d2Ref:
                        I21d2Ref = dW1(1,n*step)*dW2d2(1,n*step)/2;
                        for j=1:step-1
                            I21d2Ref = I21d2Ref + dW1(1,n*step+j)*dW2d2(1,n*step+j)/2 + dW2d2(1,n*step+j)*(W1(1,n*step+j)-W1(1,n*step));
                        end
                    % I22d2Ref:
                        I22d2Ref = dW2(1,n*step)*dW2d2(1,n*step)/2;
                        for j=1:step-1
                            I22d2Ref = I22d2Ref + dW2(1,n*step+j)*dW2d2(1,n*step+j)/2 + dW2d2(1,n*step+j)*(W2(1,n*step+j)-W2(1,n*step));
                        end
                    I21d2Ref = DelW1(1,n)*DelW2d2(1,n)-I21d2Ref;
                    I22d2Ref = DelW2(1,n)*DelW2d2(1,n)-I22d2Ref;
                end
                MilRef(:,n+1) = Yn + a*Delt + b1*DelW1(1,n) + b2*DelW2(1,n) + Lx1b1*I11Ref + Lx2b1*I21Ref + Lx1b2*I12Ref + Lx2b2*I22Ref + Lxd_one1b1*I11d1Ref + Lxd_one1b2*I12d1Ref + Lxd_one2b1*I21d1Ref + Lxd_one2b2*I22d1Ref  +  Lxd_two1b1*I11d2Ref + Lxd_two1b2*I12d2Ref + Lxd_two2b1*I21d2Ref + Lxd_two2b2*I22d2Ref;

                % MMRef:
                Yn = MMRef(:,n);
                if t(n*step) <= d1 && t(n*step) <= d2
                    Yd1 = X0; Yd1d1 = X0; Yd2 = X0; Yd2d2 = X0;
                elseif t(n*step) <= d1 && t(n*step) > d2 && t(n*step) <= 2*d2
                    Yd1 = X0; Yd1d1 = X0; Yd2 = MMRef(:,n-d2schemetimeindex); Yd2d2 = X0;
                elseif t(n*step) <= d1 && t(n*step) > 2*d2
                    Yd1 = X0; Yd1d1 = X0; Yd2 = MMRef(:,n-d2schemetimeindex); Yd2d2 = MMRef(:,n-2*d2schemetimeindex);
                elseif t(n*step) > d1 && t(n*step) <= 2*d1 && t(n*step) <= d2
                    Yd1 = MMRef(:,n-d1schemetimeindex); Yd1d1 = X0; Yd2 = X0; Yd2d2 = X0;
                elseif t(n*step) > d1 && t(n*step) <= 2*d1 && t(n*step) > d2 && t(n*step) <= 2*d2
                    Yd1 = MMRef(:,n-d1schemetimeindex); Yd1d1 = X0; Yd2 = MMRef(:,n-d2schemetimeindex); Yd2d2 = X0;
                elseif t(n*step) > d1 && t(n*step) <= 2*d1 && t(n*step) > 2*d2
                    Yd1 = MMRef(:,n-d1schemetimeindex); Yd1d1 = X0; Yd2 = MMRef(:,n-d2schemetimeindex); Yd2d2 = MMRef(:,n-2*d2schemetimeindex);
                elseif t(n*step) > 2*d1 && t(n*step) <= d2
                    Yd1 = MMRef(:,n-d1schemetimeindex); Yd1d1 = MMRef(:,n-2*d1schemetimeindex); Yd2 = X0; Yd2d2 = X0;
                elseif t(n*step) > 2*d1 && t(n*step) > d2 && t(n*step) <= 2*d2
                    Yd1 = MMRef(:,n-d1schemetimeindex); Yd1d1 = MMRef(:,n-2*d1schemetimeindex); Yd2 = MMRef(:,n-d2schemetimeindex); Yd2d2 = X0;
                elseif t(n*step) > 2*d1 && t(n*step) > 2*d2
                    Yd1 = MMRef(:,n-d1schemetimeindex); Yd1d1 = MMRef(:,n-2*d1schemetimeindex); Yd2 = MMRef(:,n-d2schemetimeindex); Yd2d2 = MMSim(:,n-2*d2schemetimeindex);
                end
                if t(n*step) <= d1+d2
                    Yd1d2 = X0; Yd2d1 = X0;
                elseif t(n*step) > d1+d2
                    Yd1d2 = MMRef(:,n-d1schemetimeindex-d2schemetimeindex); Yd2d1 = Yd1d2;
                end
                %a,b1,b2,...
                a = B0(Yn,Yd1,Yd2);
                b1 = B1(Yn,Yd1,Yd2); b1x = B1x(Yn,Yd1,Yd2);
                b2 = B2(Yn,Yd1,Yd2); b2x = B2x(Yn,Yd1,Yd2);
                I10Ref = dW1(1,n*step)*refdt/2;
                    for j=1:step-1
                        I10Ref = I10Ref + dW1(1,n*step+j)*refdt/2 + refdt*(W1(1,n*step+j)-W1(1,n*step));
                    end
                I01Ref = DelW1(1,n)*Delt - I10Ref;
                I20Ref = dW2(1,n*step)*refdt/2;
                    for j=1:step-1
                        I20Ref = I20Ref + dW2(1,n*step+j)*refdt/2 + refdt*(W2(1,n*step+j)-W2(1,n*step));
                    end
                I02Ref = DelW2(1,n)*Delt - I20Ref;
                expOmega2 = expm( A0hatDelt + A1*DelW1(1,n) + A2*DelW2(1,n) + 1/2*( LieA0A1*(I10Ref-I01Ref)+LieA0A2*(I20Ref-I02Ref)+LieA1A2*(I21Ref-I12Ref) ) );
                Hx1b1 = (b1x)*(A1*Yn+b1)-A1*b1; % This 0 is actually the Jacobian, d(b1)/dx.   j = 1, l = 1
                Hx1b2 = (b1x)*(A2*Yn+b2)-A2*b1; %  j = 1, l = 2
                Hx2b1 = (b2x)*(A1*Yn+b1)-A1*b2; %  j = 2, l = 1
                Hx2b2 = (b2x)*(A2*Yn+b2)-A2*b2; %  j = 2, l = 1
                b1d1 = B1(Yd1,Yd1d1,Yd2d1); b2d1 = B2(Yd1,Yd1d1,Yd2d1);
                Hxd_one1b1 = B1xd1(Yn,Yd1,Yd2)*(A1*Yd1+b1d1);  % j = 1; l = 1;
                Hxd_one1b2 = B1xd1(Yn,Yd1,Yd2)*(A2*Yd1+b2d1);  % j = 1; l = 2;
                Hxd_one2b1 = B2xd1(Yn,Yd1,Yd2)*(A1*Yd1+b1d1);  % j = 2; l = 1;
                Hxd_one2b2 = B2xd1(Yn,Yd1,Yd2)*(A2*Yd1+b2d1);  % j = 2; l = 2;
                b1d2 = B1(Yd2,Yd1d2,Yd2d2); b2d2 = B2(Yd2,Yd1d2,Yd2d2);
                Hxd_two1b1 = B1xd2(Yn,Yd1,Yd2)*(A1*Yd2+b1d2);  % j = 1; l = 1;        
                Hxd_two1b2 = B1xd2(Yn,Yd1,Yd2)*(A2*Yd2+b2d2);  % j = 1; l = 2;       
                Hxd_two2b1 = B2xd2(Yn,Yd1,Yd2)*(A1*Yd2+b1d2);  % j = 2; l = 1;
                Hxd_two2b2 = B2xd2(Yn,Yd1,Yd2)*(A2*Yd2+b2d2);  % j = 2; l = 2;
                MMRef(:,n+1) = expOmega2*(Yn + (a - A1*b1 - A2*b2)*Delt + b1*DelW1(1,n) + b2*DelW2(1,n) + Hx1b1*I11Ref + Hx2b1*I12Ref + Hx1b2*I21Ref + Hx2b2*I22Ref + Hxd_one1b1*I11d1Ref + Hxd_one2b1*I12d1Ref + Hxd_one1b2*I21d1Ref + Hxd_one2b2*I22d1Ref  +  Hxd_two1b1*I11d2Ref + Hxd_two2b1*I12d2Ref + Hxd_two1b2*I21d2Ref + Hxd_two2b2*I22d2Ref );
                
            end
            
        end

%% SAVING VALUES:        

        if trial == samplepath && hvalues(h) == samplehvalue
            sampletime = t(step:step:end);
            sampleEM = EM(samplecomponent,:);
            sampleMilSim = MilSim(samplecomponent,:);
            sampleMEM = MEM(samplecomponent,:);
            sampleMMSim = MMSim(samplecomponent,:);
            if useRefinedMilsteinschemes == 1
                sampleMilRef = MilRef(samplecomponent,:);
                sampleMMRef = MMRef(samplecomponent,:);
            end
        end

        for ti = 1:NO
            EMvalues(:,ti,trial,h) = EM(:,tindexobservations(ti)/step);
            MilSimvalues(:,ti,trial,h) = MilSim(:,tindexobservations(ti)/step);
            MEMvalues(:,ti,trial,h) = MEM(:,tindexobservations(ti)/step);
            MMSimvalues(:,ti,trial,h) = MMSim(:,tindexobservations(ti)/step);
            if useRefinedMilsteinschemes == 1
                MilRefvalues(:,ti,trial,h) = MilRef(:,tindexobservations(ti)/step);
                MMRefvalues(:,ti,trial,h) = MMRef(:,tindexobservations(ti)/step);
            end
        end

        for ti = 1:NO
            ErrorEM(ti,trial,h) = norm(EMvalues(:,ti,trial,h)-Xrefvalues(:,ti,trial));
            ErrorMilSim(ti,trial,h) = norm(MilSimvalues(:,ti,trial,h)-Xrefvalues(:,ti,trial));
            ErrorMEM(ti,trial,h) = norm(MEMvalues(:,ti,trial,h)-Xrefvalues(:,ti,trial));
            ErrorMMSim(ti,trial,h) = norm(MMSimvalues(:,ti,trial,h)-Xrefvalues(:,ti,trial));
            if useRefinedMilsteinschemes == 1
                ErrorMilRef(ti,trial,h) = norm(MilRefvalues(:,ti,trial,h)-Xrefvalues(:,ti,trial));
                ErrorMMRef(ti,trial,h) = norm(MMRefvalues(:,ti,trial,h)-Xrefvalues(:,ti,trial));
            end
        end

    end

end

%% ERROR CALCULATIONS

for ti = 1:NO
    for h = 1:length(hvalues)
        MSErrorEM(ti,h) = sqrt( mean( squeeze( ErrorEM(ti,:,h).^2 ) ) );
        MSErrorMilSim(ti,h) = sqrt( mean( squeeze( ErrorMilSim(ti,:,h).^2 ) ) );
        MSErrorMEM(ti,h) = sqrt( mean( squeeze( ErrorMEM(ti,:,h).^2 ) ) );
        MSErrorMMSim(ti,h) = sqrt( mean( squeeze( ErrorMMSim(ti,:,h).^2 ) ) );
        if useRefinedMilsteinschemes == 1
            MSErrorMilRef(ti,h) = sqrt( mean( squeeze( ErrorMilRef(ti,:,h).^2 ) ) );
            MSErrorMMRef(ti,h) = sqrt( mean( squeeze( ErrorMMRef(ti,:,h).^2 ) ) );
        end
    end
end

%% TRAJECTORY PLOT:

figure(NO+1)
hold on
plot([-1,0,t],[X0(samplecomponent),X0(samplecomponent),sampleXref],'k')
plot([-1,0,sampletime],[X0(samplecomponent),X0(samplecomponent),sampleEM],'r-o')
plot([-1,0,sampletime],[X0(samplecomponent),X0(samplecomponent),sampleMilSim],'-o','color',[1/3,1/3,1])
if useRefinedMilsteinschemes == 1
    plot([-1,0,sampletime],[X0(samplecomponent),X0(samplecomponent),sampleMilRef],'-square','color',[1/3,1/3,1])
end
plot([-1,0,sampletime],[X0(samplecomponent),X0(samplecomponent),sampleMEM],'-o','color',[1,1/2,0])
plot([-1,0,sampletime],[X0(samplecomponent),X0(samplecomponent),sampleMMSim],'-o','color',[0,2/3,0])
if useRefinedMilsteinschemes == 1
    plot([-1,0,sampletime],[X0(samplecomponent),X0(samplecomponent),sampleMMRef],'-square','color',[0,2/3,0])
end
if useRefinedMilsteinschemes == 0
    legend('Reference','EM','Milstein (Simple)','MEM','MM (Simple)','Interpreter','latex','FontSize',12,'location','Northwest')
elseif useRefinedMilsteinschemes == 1
    legend('Reference','EM','Milstein (Simple)','Milstein (Refined)','MEM','MM (Simple)','MM (Refined)','Interpreter','latex','FontSize',12,'location','Northwest')
end
xlabel('$t$','Interpreter','latex','FontSize',12)
ylabel('$X_1(t)$','Interpreter','latex','FontSize',12)
title('Sample Trajectory')
axis([-1,T,min(sampleXref)-0.15,max(sampleXref)+0.15])
set(gcf,'position',[200,75,1200,500])

%% ERROR GRAPHS:

if showerrorplots == 1
% Error graph (t=Ti):
for Ti = NO%1:NO
    figure(Ti)
    hold on
    referenceorderhalf = max( [MSErrorEM(Ti,1),MSErrorMEM(Ti,1),MSErrorMilSim(Ti,1)] )*sqrt(hvalues); referenceorderone = min( [MSErrorEM(Ti,1),MSErrorMEM(Ti,1),MSErrorMilSim(Ti,1),MSErrorMMSim(Ti,1)] )*hvalues;
    plot(log(hvalues)/log(2),log10(referenceorderhalf)+1/2-1/8+1/6,'m-o','Linewidth',1)
    plot(log(hvalues)/log(2),log10(referenceorderone)+1/2-1/4-1/8,'m-square','Linewidth',1)
    plot(log(hvalues)/log(2),log(MSErrorEM(Ti,:))/log(10),'r--o','Linewidth',1)
    %plot(log(hvalues)/log(2),log(MSErrorMilSim(Ti,:))/log(10),'-*','color',[1/3,1/3,1],'Linewidth',1)
    if useRefinedMilsteinschemes == 1
        plot(log(hvalues)/log(2),log(MSErrorMilRef(Ti,:))/log(10),'--square','color',[1/3,1/3,1],'Linewidth',1)
    end
    plot(log(hvalues)/log(2),log(MSErrorMEM(Ti,:))/log(10),'-o','color',[1,1/2,0],'Linewidth',1)
    %plot(log(hvalues)/log(2),log(MSErrorMMSim(Ti,:))/log(10),'-*','color',[0,2/3,0],'Linewidth',1)
    if useRefinedMilsteinschemes == 1
        plot(log(hvalues)/log(2),log(MSErrorMMRef(Ti,:))/log(10),'-square','color',[0,2/3,0],'Linewidth',1)
    end
    xticks(log(dt)/log(2):0)
    xlabel('$\log_2h$','Interpreter','latex','FontSize',16)
%     if useRefinedMilsteinschemes == 0
%         legend('Reference $1/2$','Reference $1$','EM','Milstein (Simple)','MEM','MM (Simple)','Interpreter','latex','FontSize',14,'location','Southeast','NumColumns',1)
%     else
%         legend('Reference $1/2$','Reference $1$','EM','Milstein (Simple)','Milstein (Refined)','MEM','MM (Simple)','MM (Refined)','Interpreter','latex','FontSize',14,'location','Southeast','NumColumns',1)
%     end
    legend('Reference $1/2$','Reference $1$','EM','Milstein','MEM','MM','Interpreter','latex','FontSize',14,'location','Southeast','NumColumns',1)
    %axis([log(dt)/log(2),log(1)/log(2),axisbounds(1,Ti)-0.5,axisbounds(2,Ti)+0.25])
    xlim([log(dt)/log(2)-0.1,max(log(hvalues)/log(2))+0.1])
    ylabel('$\log_{10}MSE(4)$','Interpreter','latex','FontSize',16)
    %title(['MS Strong Errors ','($t=$ ',num2str(t(tindexobservations(Ti))),')'],'Interpreter','latex','FontSize',16)
    set(gcf,'position',[200,75,700,500])
    grid on
    a = get(gca,'XTickLabel');
    set(gca,'XTickLabel',a,'fontsize',12)
    b = get(gca,'YTickLabel');
    set(gca,'YTickLabel',b,'fontsize',12)

    if saveplots == 1 && d1 == 1 && d2 == 1/4 && runs == 10000 && Ti == NO
        saveas(gcf,'ZExample3.eps','epsc')
    end
end
end
disp(['Total simulation time is ',num2str(round(toc(simulationtime)/60,4)),' minutes.'])