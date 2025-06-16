% This script is for plotting the mean-square error graphs of the linear,
% homogeneous, d-dimensional SDDE
%  dX(t) = \sum_{j=0}^m ( A_j*X(t) + B_j(X(t-1)) ) dW_j(t), 0<=t<=T,
% X(0)=X0, X(t) = X(0) for t<0,   where m = 2 and d = 2.

%% USER PARAMETERS:
savefigures = 0; % Set this to 1 to save figures, and 0 when not saving figures.
runs = 5000;     % Number of trajectories.

%% SIMULATION BEGINNING:

dt = 2^-10;    % Smallest time increment in the approximations.
refdt = 2^-14; % Step size in reference solution.
% Equation parameters:
X0 = [0.8;0.2];
A0 = [-0.1,0.4;-0.3,0.2];   A1 = [0.3,0.1;0,0.2];  A2 = [0.1,0;0.3,0.1];
B0 = @(Xd) [cos(Xd(1)+Xd(2));Xd(2)-Xd(1)^2]/10;
B1 = @(Xd) [sin(Xd(1))+exp(-Xd(2)^2);atan(Xd(1))+cos(Xd(2))]/3;
B2 = @(Xd) [0.18,0.04;0.21,0.03]*[Xd(1);atan(Xd(2))/4];

useRefinedMilsteinschemes = 1;  % This is hardcoded.

T = 6;  % T > 1 is required in this script.
        
% Observation parameters:
hvalues = 2.^(-0:-1:log(dt)/log(2));        % Set of step sizes to look at.
samplepath = 1;                             % Sample trajectory to view.
samplecomponent = 1;                        % Sample component to view.
samplehvalue = 2^-4;                        % Sample step size to view.

%% CALCULATIONS BEGIN HERE.

simulationtime = tic;

% Calculate Jacobians:
syms y1 y2
    B1xdtemp = eval(['@(y1,y2)' char(jacobian(B1([y1;y2]),[y1;y2]))]);
    B1xd = @(Y) B1xdtemp(Y(1),Y(2));
    B2xdtemp = eval(['@(y1,y2)' char(jacobian(B2([y1;y2]),[y1;y2]))]);
    B2xd = @(Y) B2xdtemp(Y(1),Y(2));
clear y1 y2

%% INITIALISING CALCULATIONS:

rng('default');close all
T = ceil(T);    % This is ensuring T is an integer.

d = 2;  t = refdt:refdt:T;
tindex = (1:T)/refdt;

A0hat = A0 - (A1^2+A2^2)/2;

Xrefvalues = zeros(d,T,runs);

LieA0A1 = A0*A1-A1*A0;
LieA0A2 = A0*A2-A2*A0;
LieA1A2 = A1*A2-A2*A1;

% Initialise the errors, and compile them in vectors, so that we can view
% their means plotted against the step sizes, at the end of this script.
% The first variable refers to the time that the error is observed at. For
% example, ErrorEM(1,.,.) is the error at t = 1, while ErrorEM(2,.,.) is the
% error at t = 2.
ErrorEM = ones(T,runs,length(hvalues));     % Strong errors compared with X, a reference solution formed by the analytic expression with time step refdt.
ErrorMilSim = ones(T,runs,length(hvalues));
ErrorMEM = ones(T,runs,length(hvalues));
ErrorMMSim = ones(T,runs,length(hvalues));
if useRefinedMilsteinschemes == 1
    ErrorMilRef = ones(T,runs,length(hvalues));    
    ErrorMMRef = ones(T,runs,length(hvalues));
end

% Stacks for the values:
EMvalues = zeros(d,T,runs,length(hvalues));  % Indexing is time, then trial, then step size for the scheme.
MilSimvalues = zeros(d,T,runs,length(hvalues));
MEMvalues = zeros(d,T,runs,length(hvalues));
MMSimvalues = zeros(d,T,runs,length(hvalues));
if useRefinedMilsteinschemes == 1
    MilRefvalues = zeros(d,T,runs,length(hvalues));
    MMRefvalues = zeros(d,T,runs,length(hvalues));
end

MSErrorEM = ones(T,length(hvalues));
MSErrorMilSim = ones(T,length(hvalues));
MSErrorMEM = ones(T,length(hvalues));
MSErrorMMSim = ones(T,length(hvalues));
if useRefinedMilsteinschemes == 1
    MSErrorMilRef = ones(T,length(hvalues));
    MSErrorMMRef = ones(T,length(hvalues));
end

%% SIMULATIONS BEGIN HERE:

for trial = 1:runs

%% SIMULATION OF WIENER PATH:

    % W1 and W1d, beginning with W1:
    sqrtrefdt = sqrt(refdt);                    % Value used in the following calculation.
        dW1 = sqrtrefdt*randn(1,tindex(T));    % Brownian increments.  This is set up so that dW(1,n) = W(1,n+1) - W(1,n),  like DeltaW often is.
        dW1initial = sqrtrefdt*randn(1,1);
        W1initial = 0;
        W1 = zeros(1,tindex(T));
        for n = 1:tindex(T)
            if n == 1
                W1(1,n) = dW1initial(1,1);
            else
                W1(1,n) = W1(1,n-1) + dW1(1,n-1);
            end
        end
        % Define W1d(t) = W1(t-1):
        W1d = zeros(1,tindex(T));
        dW1d = zeros(1,tindex(T));
        W1d(1,tindex(1)) = W1initial(1,1);
        for i = 1:tindex(T-1)
            W1d(1,tindex(1)+i) = W1(1,i);
        end
        for n = tindex(1):tindex(T)-1
            dW1d(1,n) = W1d(1,n+1)-W1d(1,n);
        end
    % W2 and W2d:
    sqrtrefdt = sqrt(refdt);                    % Value used in the following calculation.
        dW2 = sqrtrefdt*randn(1,tindex(T));    % Brownian increments.  This is set up so that dW(1,n) = W(1,n+1) - W(1,n),  like DeltaW often is.
        dW2initial = sqrtrefdt*randn(1,1);
        W2initial = 0;
        W2 = zeros(1,tindex(T));
        for n = 1:tindex(T)
            if n == 1
                W2(1,n) = dW2initial(1,1);
            else
                W2(1,n) = W2(1,n-1) + dW2(1,n-1);
            end
        end
        % Define W2d(t) = W2(t-1):
        W2d = zeros(1,tindex(T));
        dW2d = zeros(1,tindex(T));
        W2d(1,tindex(1)) = W2initial(1,1);
        for i = 1:tindex(T-1)
            W2d(1,tindex(1)+i) = W2(1,i);
        end
        for n = tindex(1):tindex(T)-1
            dW2d(1,n) = W2d(1,n+1)-W2d(1,n);
        end

%% SIMULATION OF REFERENCE SOLUTION:

    % Construct the reference (analytic) solution (with step size h = refdt):
    Xref = zeros(d,tindex(T));
    % Initial step:
    Xn = X0; Xd = X0;
    a = A0*Xn + B0(Xd);
    b1 = A1*Xn + B1(Xd);  b1x = A1;
    b2 = A2*Xn + B2(Xd);  b2x = A2;
    I11 = (dW1initial(1,1)^2-refdt)/2; I22 = (dW2initial(1,1)^2-refdt)/2;
        % I12ref calculation:
        I12 = dW1initial(1,1)*dW2initial(1,1)/2;
        I21 = dW1initial(1,1)*dW2initial(1,1) - I12;
    % Update these Iijs for these refined ones.
    Lx1b1 = (b1x)*b1;   Lx2b2 = (b2x)*b2;   Lx1b2 = (b2x)*b1;   Lx2b1 = (b1x)*b2;
    Xref(:,1) = Xn + a*refdt + b1*dW1initial(1,1) + b2*dW2initial(1,1) + Lx1b1*I11 + Lx2b1*I21 + Lx1b2*I12 + Lx2b2*I22;
    % First time interval until first delay point:
    for n = 1:tindex(1)
        Xn = Xref(:,n); Xd = X0;
        a = A0*Xn + B0(Xd);
        b1 = A1*Xn + B1(Xd);  b1x = A1;
        b2 = A2*Xn + B2(Xd);  b2x = A2;
        I11 = (dW1(1,n)^2-refdt)/2; I22 = (dW2(1,n)^2-refdt)/2;
            % I12ref calculation:
            I12 = dW1(1,n)*dW2(1,n)/2;
            I21 = dW1(1,n)*dW2(1,n) - I12;
        Lx1b1 = (b1x)*b1;   Lx2b2 = (b2x)*b2;   Lx1b2 = (b2x)*b1;   Lx2b1 = (b1x)*b2;
        Xref(:,n+1) = Xn + a*refdt + b1*dW1(1,n) + b2*dW2(1,n) + Lx1b1*I11 + Lx2b1*I21 + Lx1b2*I12 + Lx2b2*I22;
    end
    % Successive intervals between delays:
    for Ti = 2:T
        for n = tindex(Ti-1)+1:tindex(Ti)

            % MilRef:
            Xn = Xref(:,n);
            if n == tindex(1)
                Xd = X0;
            else
                Xd = Xref(:,n-tindex(1));
            end
            if n <= tindex(2)
                Xd2 = X0;
            else
                Xd2 = Xref(:,n-tindex(2));
            end
            a = A0*Xn + B0(Xd);
            b1 = A1*Xn + B1(Xd);  b1x = A1;
            b2 = A2*Xn + B2(Xd);  b2x = A2;
            I11 = (dW1(1,n)^2-refdt)/2; I22 = (dW2(1,n)^2-refdt)/2;
                % I12ref calculation:
                I12 = dW1(1,n)*dW2(1,n)/2;
                I21 = dW1(1,n)*dW2(1,n) - I12;
            Lx1b1 = (b1x)*b1;   Lx2b2 = (b2x)*b2;   Lx1b2 = (b2x)*b1;   Lx2b1 = (b1x)*b2;
            b1xd = B1xd(Xd); b2xd = B2xd(Xd); b1d = A1*Xd + B1(Xd2); b2d = A2*Xd + B2(Xd2);
            Lxd1b1 = (b1xd)*b1d; Lxd2b2 = (b2xd)*b2d;   Lxd1b2 = (b2xd)*b1d;   Lxd2b1 = (b1xd)*b2d;
            % Iijd calculations:
                    % I11dRef:
                        I11d = dW1d(1,n)*dW1(1,n)/2;
                    % I12dRef:
                        I12d = dW1d(1,n)*dW2(1,n)/2;
                    % I21dRef:
                        I21d = dW1(1,n)*dW2d(1,n)/2;
                    % I22dRef:
                        I22d = dW2(1,n)*dW2d(1,n)/2;
                if n < tindex(T) % This condition is because Milstein defines Xref(:,n+1) based on Xref(:,n), and we want to stop at time T. (*)
                    Xref(:,n+1) = Xn + a*refdt + b1*dW1(1,n) + b2*dW2(1,n) + Lx1b1*I11 + Lx2b1*I21 + Lx1b2*I12 + Lx2b2*I22 + Lxd1b1*I11d + Lxd1b2*I12d + Lxd2b1*I21d + Lxd2b2*I22d;
                end
        end
    end
    %Xref = Xref(:,1:end-1); A condition such as this is needed it we don't have the calculation as we do at (*), just above, and allow Xref to be defined beyond T.
    for ti = 1:T
        Xrefvalues(:,ti,trial) = Xref(:,ti/refdt);
    end
    if trial == samplepath
        sampleXref = Xref(samplecomponent,:);
    end
        
%% SCHEMES:

    for h = 1:length(hvalues)

        % Define some constants that are used throughout the following part:
        step = hvalues(h)/refdt;    % This is the index multiplier, for how many index positions comprise a single step of the simulations.  The idea is that t_n = t(n*step).
        N = tindex(T)/step;           % This is the number of steps that are made in [0,T].          
        Delt = hvalues(h);          % Step size.                 
        A0hatDelt = (A0 - A1^2/2 - A2^2/2)*Delt;

        % DelW1 and DelW1d:
        DelW1 = zeros(1,N-1);  % Define Delta W1 (see page 340 of Kloeden and Platen).
        DelW1initial = zeros(1,1);
        DelW1d = zeros(1,N-1);
        DelW1dinitial = zeros(1,1);
        DelW1initial(1,1) = W1(1,step);
        for n = 1:N-1
            DelW1(1,n) = W1(1,(n+1)*step) - W1(1,n*step);
            if n == N/T
                DelW1dinitial(1,1) = DelW1initial(1,1);
                DelW1d(1,n) = DelW1dinitial(1,1);
            elseif n > N/T
                DelW1d(1,n) = DelW1(1,n-N/T);
            end
        end

        % DelW2 and DelW2d:
        DelW2 = zeros(1,N-1);  % Define Delta W1 (see page 340 of Kloeden and Platen).
        DelW2initial = zeros(1,1);
        DelW2d = zeros(1,N-1);
        DelW2dinitial = zeros(1,1);
        DelW2initial(1,1) = W2(1,step);
        for n = 1:N-1
            DelW2(1,n) = W2(1,(n+1)*step) - W2(1,n*step);
            if n == N/T
                DelW2dinitial(1,1) = DelW2initial(1,1);
                DelW2d(1,n) = DelW2dinitial(1,1);
            elseif n > N/T
                DelW2d(1,n) = DelW2(1,n-N/T);
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

        % EM:
        Yn = X0; Yd = X0;
        a = A0*Yn + B0(Yd);
        b1 = A1*Yn + B1(Yd);
        b2 = A2*Yn + B2(Yd);
        EM(:,1) = X0 + a*Delt + b1*DelW1initial(1,1) + b2*DelW2initial(1,1);
        
        % MilSim:
        Yn = X0; Yd = X0;
        a = A0*Yn + B0(Yd);
        b1 = A1*Yn + B1(Yd); b1x = A1;
        b2 = A2*Yn + B2(Yd); b2x = A2;
        I11Sim = (DelW1initial(1,1)^2-Delt)/2; I22Sim = (DelW2initial(1,1)^2-Delt)/2; I12Sim = DelW1initial(1,1)*DelW2initial(1,1)/2; I21Sim = DelW1initial(1,1)*DelW2initial(1,1)/2;
        Lx1b1 = (b1x)*b1;   Lx2b2 = (b2x)*b2;   Lx1b2 = (b2x)*b1;   Lx2b1 = (b1x)*b2;
        MilSim(:,1) = Yn + a*Delt + b1*DelW1initial(1,1) + b2*DelW2initial(1,1) + Lx1b1*I11Sim + Lx2b1*I21Sim + Lx1b2*I12Sim + Lx2b2*I22Sim;
        
        % MEM:
        Yn = X0; Yd = X0;
        a = B0(Yd);
        b1 = B1(Yd);
        b2 = B2(Yd);
        expOmega1 = expm( A0hatDelt + A1*DelW1initial(1,1) + A2*DelW2initial(1,1) );
        MEM(:,1) = expOmega1*(Yn + (a - A1*b1 - A2*b2)*Delt + b1*DelW1initial(1,1) + b2*DelW2initial(1,1) );
        
        % MMSim:
        Yn = X0; Yd = X0;
        a = B0(Yd);
        b1 = B1(Yd);
        b2 = B2(Yd);
        I10Sim = DelW1initial(1,1)*Delt/2; I01Sim = DelW1initial(1,1)*Delt - I10Sim; I20Sim = DelW2initial(1,1)*Delt/2; I02Sim = DelW2initial(1,1)*Delt - I20Sim;
        expOmega2 = expm( A0hatDelt + A1*DelW1initial(1,1) + A2*DelW2initial(1,1) + 1/2*( LieA0A1*(I10Sim-I01Sim)+LieA0A2*(I20Sim-I02Sim)+LieA1A2*(I21Sim-I12Sim) ) );
        Hx1b1 = (0)*(A1*Yn+b1)-A1*b1; % This 0 is actually the Jacobian, d(b1)/dx.   j = 1, l = 1
        Hx1b2 = (0)*(A2*Yn+b2)-A2*b1; %  j = 1, l = 2
        Hx2b1 = (0)*(A1*Yn+b1)-A1*b2; %  j = 2, l = 1
        Hx2b2 = (0)*(A2*Yn+b2)-A2*b2; %  j = 2, l = 1
        MMSim(:,1) = expOmega2*(Yn + (a - A1*b1 - A2*b2)*Delt + b1*DelW1initial(1,1) + b2*DelW2initial(1,1) + Hx1b1*I11Sim + Hx1b2*I21Sim + Hx2b1*I12Sim + Hx2b2*I22Sim );
        
        if useRefinedMilsteinschemes == 1
            
            % MilRef:
            Yn = X0; Yd = X0;
            a = A0*Yn + B0(Yd);
            b1 = A1*Yn + B1(Yd);  b1x = A1;
            b2 = A2*Yn + B2(Yd);  b2x = A2;
            I11Ref = (DelW1initial(1,1)^2-Delt)/2; I22Ref = (DelW2initial(1,1)^2-Delt)/2;
                % I12ref calculation:
                I12Ref = dW1initial(1,1)*dW2initial(1,1)/2;
                for j=1:step-1
                    I12Ref = I12Ref + dW1(1,j)*dW2(1,j)/2 + dW2(1,j)*(W1(1,j)-W1initial(1,1));
                end
                I21Ref = DelW1initial(1,1)*DelW2initial(1,1) - I12Ref;
            % Update these Iijs for these refined ones.
            Lx1b1 = (b1x)*b1;   Lx2b2 = (b2x)*b2;   Lx1b2 = (b2x)*b1;   Lx2b1 = (b1x)*b2;
            MilRef(:,1) = Yn + a*Delt + b1*DelW1initial(1,1) + b2*DelW2initial(1,1) + Lx1b1*I11Ref + Lx2b1*I21Ref + Lx1b2*I12Ref + Lx2b2*I22Ref;
            
            % MMRef:
            Yn = X0; Yd = X0;
            a = B0(Yd);
            b1 = B1(Yd);
            b2 = B2(Yd);
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
            Hx1b1 = (0)*(A1*Yn+b1)-A1*b1; % This 0 is actually the Jacobian, d(b1)/dx.   j = 1, l = 1
            Hx1b2 = (0)*(A2*Yn+b2)-A2*b1; %  j = 1, l = 2
            Hx2b1 = (0)*(A1*Yn+b1)-A1*b2; %  j = 2, l = 1
            Hx2b2 = (0)*(A2*Yn+b2)-A2*b2; %  j = 2, l = 1
            MMRef(:,1) = expOmega2*(Yn + (a - A1*b1 - A2*b2)*Delt + b1*DelW1initial(1,1) + b2*DelW2initial(1,1) + Hx1b1*I11Ref + Hx1b2*I21Ref + Hx2b1*I12Ref + Hx2b2*I22Ref );
            
        end

%% STEP ONE:

        % Beginning of scheme calculation from time h to tau.
        for n = 1 : N/T - 1
            
            % EM:
            Yn = EM(:,n); Yd = X0;
            a = A0*Yn + B0(Yd);
            b1 = A1*Yn + B1(Yd);
            b2 = A2*Yn + B2(Yd);
            EM(:,n+1) = Yn + a*Delt + b1*DelW1(1,n) + b2*DelW2(1,n);
             
            % MilSim:
            Yn = MilSim(:,n); Yd = X0;
            a = A0*Yn + B0(Yd);
            b1 = A1*Yn + B1(Yd); b1x = A1;
            b2 = A2*Yn + B2(Yd); b2x = A2;
            I11Sim = (DelW1(1,n)^2-Delt)/2; I22Sim = (DelW2(1,n)^2-Delt)/2; I12Sim = DelW1(1,n)*DelW2(1,n)/2; I21Sim = DelW1(1,n)*DelW2(1,n)/2;
            Lx1b1 = (b1x)*b1;   Lx2b2 = (b2x)*b2;   Lx1b2 = (b2x)*b1;   Lx2b1 = (b1x)*b2;
            MilSim(:,n+1) = Yn + a*Delt + b1*DelW1(1,n) + b2*DelW2(1,n) + Lx1b1*I11Sim + Lx2b1*I21Sim + Lx1b2*I12Sim + Lx2b2*I22Sim;
            
            % MEM:
            Yn = MEM(:,n); Yd = X0;
            a = B0(Yd);
            b1 = B1(Yd);
            b2 = B2(Yd);
            expOmega1 = expm( A0hatDelt + A1*DelW1(1,n) + A2*DelW2(1,n) );
            MEM(:,n+1) = expOmega1*(Yn + (a - A1*b1 - A2*b2)*Delt + b1*DelW1(1,n) + b2*DelW2(1,n) );
            
            % MMSim:
            Yn = MMSim(:,n); Yd = X0;
            a = B0(Yd);
            b1 = B1(Yd);
            b2 = B2(Yd);
            I10Sim = DelW1(1,n)*Delt/2; I01Sim = DelW1(1,n)*Delt - I10Sim; I20Sim = DelW2(1,n)*Delt/2; I02Sim = DelW2(1,n)*Delt - I20Sim;
            expOmega2 = expm( A0hatDelt + A1*DelW1(1,n) + A2*DelW2(1,n) + 1/2*( LieA0A1*(I10Sim-I01Sim)+LieA0A2*(I20Sim-I02Sim)+LieA1A2*(I21Sim-I12Sim) ) );
            Hx1b1 = (0)*(A1*Yn+b1)-A1*b1; % This 0 is actually the Jacobian, d(b1)/dx.   j = 1, l = 1
            Hx1b2 = (0)*(A2*Yn+b2)-A2*b1; %  j = 1, l = 2
            Hx2b1 = (0)*(A1*Yn+b1)-A1*b2; %  j = 2, l = 1
            Hx2b2 = (0)*(A2*Yn+b2)-A2*b2; %  j = 2, l = 1
            MMSim(:,n+1) = expOmega2*(Yn + (a - A1*b1 - A2*b2)*Delt + b1*DelW1(1,n) + b2*DelW2(1,n) + Hx1b1*I11Sim + Hx1b2*I21Sim + Hx2b1*I12Sim + Hx2b2*I22Sim );

            if useRefinedMilsteinschemes == 1
                
                % MilRef:
                Yn = MilRef(:,n); Yd = X0;
                a = A0*Yn + B0(Yd);
                b1 = A1*Yn + B1(Yd);  b1x = A1;
                b2 = A2*Yn + B2(Yd);  b2x = A2;
                I11Ref = (DelW1(1,n)^2-Delt)/2; I22Ref = (DelW2(1,n)^2-Delt)/2;
                    % I12ref calculation:
                    I12Ref = dW1(1,n*step)*dW2(1,n*step)/2;
                    for j=1:step-1
                        I12Ref = I12Ref + dW1(1,n*step+j)*dW2(1,n*step+j)/2 + dW2(1,n*step+j)*(W1(1,n*step+j)-W1(1,n*step));
                    end
                    I21Ref = DelW1(1,n)*DelW2(1,n) - I12Ref;
                Lx1b1 = (b1x)*b1;   Lx2b2 = (b2x)*b2;   Lx1b2 = (b2x)*b1;   Lx2b1 = (b1x)*b2;
                MilRef(:,n+1) = Yn + a*Delt + b1*DelW1(1,n) + b2*DelW2(1,n) + Lx1b1*I11Ref + Lx2b1*I21Ref + Lx1b2*I12Ref + Lx2b2*I22Ref;
  
                % MMRef:
                Yn = MMRef(:,n); Yd = X0;
                a = B0(Yd);
                b1 = B1(Yd);
                b2 = B2(Yd);
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
                Hx1b1 = (0)*(A1*Yn+b1)-A1*b1; % This 0 is actually the Jacobian, d(b1)/dx.   j = 1, l = 1
                Hx1b2 = (0)*(A2*Yn+b2)-A2*b1; %  j = 1, l = 2
                Hx2b1 = (0)*(A1*Yn+b1)-A1*b2; %  j = 2, l = 1
                Hx2b2 = (0)*(A2*Yn+b2)-A2*b2; %  j = 2, l = 1
                MMRef(:,n+1) = expOmega2*(Yn + (a - A1*b1 - A2*b2)*Delt + b1*DelW1(1,n) + b2*DelW2(1,n) + Hx1b1*I11Ref + Hx1b2*I21Ref + Hx2b1*I12Ref + Hx2b2*I22Ref );
                                 
            end
        end
        % End of scheme calculation from time h to tau.

%% STEP TWO:

        % Beginning for time tau <= t < 2*tau.
        for n = N/T:2*N/T-1
            
            % EM:
            Yn = EM(:,n);
            if n == N/T
                Yd = X0;
            else
                Yd = EM(:,n-N/T);
            end       
            a = A0*Yn + B0(Yd);
            b1 = A1*Yn + B1(Yd);
            b2 = A2*Yn + B2(Yd);
            EM(:,n+1) = Yn + a*Delt + b1*DelW1(1,n) + b2*DelW2(1,n);
                            
            % MilSim:
            Yn = MilSim(:,n); Yd2 = X0;
            if n == N/T
                Yd = X0;
            else
                Yd = MilSim(:,n-N/T);
            end  
            a = A0*Yn + B0(Yd);
            b1 = A1*Yn + B1(Yd); b1x = A1;
            b2 = A2*Yn + B2(Yd); b2x = A2;
            I11Sim = (DelW1(1,n)^2-Delt)/2; I22Sim = (DelW2(1,n)^2-Delt)/2; I12Sim = DelW1(1,n)*DelW2(1,n)/2; I21Sim = DelW1(1,n)*DelW2(1,n)/2;
            Lx1b1 = (b1x)*b1;   Lx2b2 = (b2x)*b2;   Lx1b2 = (b2x)*b1;   Lx2b1 = (b1x)*b2;
            b1xd = B1xd(Yd); b2xd = B2xd(Yd); b1d = A1*Yd + B1(Yd2); b2d = A2*Yd + B2(Yd2);
            Lxd1b1 = (b1xd)*b1d; Lxd2b2 = (b2xd)*b2d;   Lxd1b2 = (b2xd)*b1d;   Lxd2b1 = (b1xd)*b2d;
            I11dSim = DelW1d(1,n)*DelW1(1,n)/2; I22dSim = DelW2d(1,n)*DelW2(1,n)/2; I12dSim = DelW1d(1,n)*DelW2(1,n)/2; I21dSim = DelW2d(1,n)*DelW1(1,n)/2;
            MilSim(:,n+1) = Yn + a*Delt + b1*DelW1(1,n) + b2*DelW2(1,n) + Lx1b1*I11Sim + Lx1b2*I12Sim + Lx2b1*I21Sim + Lx2b2*I22Sim + Lxd1b1*I11dSim + Lxd1b2*I12dSim + Lxd2b1*I21dSim + Lxd2b2*I22dSim;
            
            % MEM:
            Yn = MEM(:,n); 
            if n == N/T
                Yd = X0;
            else
                Yd = MEM(:,n-N/T);
            end 
            a = B0(Yd);
            b1 = B1(Yd);
            b2 = B2(Yd);
            expOmega1 = expm( A0hatDelt + A1*DelW1(1,n) + A2*DelW2(1,n) );
            MEM(:,n+1) = expOmega1*(Yn + (a - A1*b1 - A2*b2)*Delt + b1*DelW1(1,n) + b2*DelW2(1,n) );
            
            % MMSim:
            Yn = MMSim(:,n); Yd2 = X0;
            if n == N/T
                Yd = X0;
            else
                Yd = MMSim(:,n-N/T);
            end 
            a = B0(Yd);
            b1 = B1(Yd);
            b2 = B2(Yd);
            I10Sim = DelW1(1,n)*Delt/2; I01Sim = DelW1(1,n)*Delt - I10Sim; I20Sim = DelW2(1,n)*Delt/2; I02Sim = DelW2(1,n)*Delt - I20Sim;
            expOmega2 = expm( A0hatDelt + A1*DelW1(1,n) + A2*DelW2(1,n) + 1/2*( LieA0A1*(I10Sim-I01Sim)+LieA0A2*(I20Sim-I02Sim)+LieA1A2*(I21Sim-I12Sim) ) );
            Hx1b1 = (0)*(A1*Yn+b1)-A1*b1; % This 0 is actually the Jacobian, d(b1)/dx.   j = 1, l = 1
            Hx1b2 = (0)*(A2*Yn+b2)-A2*b1; %  j = 1, l = 2
            Hx2b1 = (0)*(A1*Yn+b1)-A1*b2; %  j = 2, l = 1
            Hx2b2 = (0)*(A2*Yn+b2)-A2*b2; %  j = 2, l = 1
            b1d = B1(Yd2); b2d = B2(Yd2);
            b1xd = B1xd(Yd); b2xd = B2xd(Yd);
            Hxd1b1 = b1xd*(A1*Yd+b1d);  % j = 1; l = 1;        
            Hxd1b2 = b1xd*(A2*Yd+b2d);  % j = 1; l = 2;        
            Hxd2b1 = b2xd*(A1*Yd+b1d);  % j = 2; l = 1;
            Hxd2b2 = b2xd*(A2*Yd+b2d);  % j = 2; l = 2;
            MMSim(:,n+1) = expOmega2*(Yn + (a - A1*b1 - A2*b2)*Delt + b1*DelW1(1,n) + b2*DelW2(1,n) + Hx1b1*I11Sim + Hx1b2*I21Sim + Hx2b1*I12Sim + Hx2b2*I22Sim + Hxd1b1*I11dSim + Hxd2b1*I12dSim + Hxd1b2*I21dSim + Hxd2b2*I22dSim );

            if useRefinedMilsteinschemes == 1
                
                % MilRef:
                Yn = MilRef(:,n); Yd2 = X0;
                if n == N/T
                    Yd = X0;
                else
                    Yd = MilRef(:,n-N/T);
                end  
                a = A0*Yn + B0(Yd);
                b1 = A1*Yn + B1(Yd);  b1x = A1;
                b2 = A2*Yn + B2(Yd);  b2x = A2;
                I11Ref = (DelW1(1,n)^2-Delt)/2; I22Ref = (DelW2(1,n)^2-Delt)/2;
                    % I12ref calculation:
                    I12Ref = dW1(1,n*step)*dW2(1,n*step)/2;
                    for j=1:step-1
                        I12Ref = I12Ref + dW1(1,n*step+j)*dW2(1,n*step+j)/2 + dW2(1,n*step+j)*(W1(1,n*step+j)-W1(1,n*step));
                    end
                    I21Ref = DelW1(1,n)*DelW2(1,n) - I12Ref;
                Lx1b1 = (b1x)*b1;   Lx2b2 = (b2x)*b2;   Lx1b2 = (b2x)*b1;   Lx2b1 = (b1x)*b2;
                b1xd = B1xd(Yd); b2xd = B2xd(Yd); b1d = A1*Yd + B1(Yd2); b2d = A2*Yd + B2(Yd2);
                Lxd1b1 = (b1xd)*b1d; Lxd2b2 = (b2xd)*b2d;   Lxd1b2 = (b2xd)*b1d;   Lxd2b1 = (b1xd)*b2d;
                % IijdRef calculations:
                        % I11dRef:
                            I11dRef = dW1d(1,n*step)*dW1(1,n*step)/2;
                            for j=1:step-1
                                I11dRef = I11dRef + dW1d(1,n*step+j)*dW1(1,n*step+j)/2 + dW1(1,n*step+j)*(W1d(1,n*step+j)-W1d(1,n*step));
                            end
                        % I12dRef:
                            I12dRef = dW1d(1,n*step)*dW2(1,n*step)/2;
                            for j=1:step-1
                                I12dRef = I12dRef + dW1d(1,n*step+j)*dW2(1,n*step+j)/2 + dW2(1,n*step+j)*(W1d(1,n*step+j)-W1d(1,n*step));
                            end
                        % I21dRef:
                            I21dRef = dW1(1,n*step)*dW2d(1,n*step)/2;
                            for j=1:step-1
                                I21dRef = I21dRef + dW1(1,n*step+j)*dW2d(1,n*step+j)/2 + dW2d(1,n*step+j)*(W1(1,n*step+j)-W1(1,n*step));
                            end
                        % I22dRef:
                            I22dRef = dW2(1,n*step)*dW2d(1,n*step)/2;
                            for j=1:step-1
                                I22dRef = I22dRef + dW2(1,n*step+j)*dW2d(1,n*step+j)/2 + dW2d(1,n*step+j)*(W2(1,n*step+j)-W2(1,n*step));
                            end
                I21dRef = DelW1(1,n)*DelW2d(1,n)-I21dRef;
                I22dRef = DelW2(1,n)*DelW2d(1,n)-I22dRef;    
                MilRef(:,n+1) = Yn + a*Delt + b1*DelW1(1,n) + b2*DelW2(1,n) + Lx1b1*I11Ref + Lx2b1*I21Ref + Lx1b2*I12Ref + Lx2b2*I22Ref + Lxd1b1*I11dRef + Lxd1b2*I12dRef + Lxd2b1*I21dRef + Lxd2b2*I22dRef;
                
                % MMRef:
                Yn = MMRef(:,n); Yd2 = X0;
                if n == N/T
                    Yd = X0;
                else
                    Yd = MMRef(:,n-N/T);
                end 
                a = B0(Yd);
                b1 = B1(Yd);
                b2 = B2(Yd);
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
                Hx1b1 = (0)*(A1*Yn+b1)-A1*b1; % This 0 is actually the Jacobian, d(b1)/dx.   j = 1, l = 1
                Hx1b2 = (0)*(A2*Yn+b2)-A2*b1; %  j = 1, l = 2
                Hx2b1 = (0)*(A1*Yn+b1)-A1*b2; %  j = 2, l = 1
                Hx2b2 = (0)*(A2*Yn+b2)-A2*b2; %  j = 2, l = 1
                b1d = B1(Yd2); b2d = B2(Yd2);
                b1xd = B1xd(Yd); b2xd = B2xd(Yd);
                Hxd1b1 = b1xd*(A1*Yd+b1d);  % j = 1; l = 1;        
                Hxd1b2 = b1xd*(A2*Yd+b2d);  % j = 1; l = 2;        
                Hxd2b1 = b2xd*(A1*Yd+b1d);  % j = 2; l = 1;
                Hxd2b2 = b2xd*(A2*Yd+b2d);  % j = 2; l = 2;
                MMRef(:,n+1) = expOmega2*(Yn + (a - A1*b1 - A2*b2)*Delt + b1*DelW1(1,n) + b2*DelW2(1,n) + Hx1b1*I11Ref + Hx2b1*I12Ref + Hx1b2*I21Ref + Hx2b2*I22Ref + Hxd1b1*I11dRef + Hxd2b1*I12dRef + Hxd1b2*I21dRef + Hxd2b2*I22dRef );

            end
        end
        % End for time tau <= t < 2*tau.

%% STEP THREE:

        % Beginning for time 2*tau <= t < 3*tau.
        if T > 2
            for n = 2*N/T : 3*N/T-1
                
                % EM:
                Yn = EM(:,n);
                    Yd = EM(:,n-N/T); 
                a = A0*Yn + B0(Yd);
                b1 = A1*Yn + B1(Yd);
                b2 = A2*Yn + B2(Yd);
                EM(:,n+1) = Yn + a*Delt + b1*DelW1(1,n) + b2*DelW2(1,n);
                                
                % MilSim:
                Yn = MilSim(:,n);
                    Yd = MilSim(:,n-N/T);
                if n == 2*N/T
                    Yd2 = X0;
                else
                    Yd2 = MilSim(:,n-2*N/T);
                end  
                a = A0*Yn + B0(Yd);
                b1 = A1*Yn + B1(Yd); b1x = A1;
                b2 = A2*Yn + B2(Yd); b2x = A2;
                I11Sim = (DelW1(1,n)^2-Delt)/2; I22Sim = (DelW2(1,n)^2-Delt)/2; I12Sim = DelW1(1,n)*DelW2(1,n)/2; I21Sim = DelW1(1,n)*DelW2(1,n)/2;
                Lx1b1 = (b1x)*b1;   Lx2b2 = (b2x)*b2;   Lx1b2 = (b2x)*b1;   Lx2b1 = (b1x)*b2;
                b1xd = B1xd(Yd); b2xd = B2xd(Yd); b1d = A1*Yd + B1(Yd2); b2d = A2*Yd + B2(Yd2);
                Lxd1b1 = (b1xd)*b1d; Lxd2b2 = (b2xd)*b2d;   Lxd1b2 = (b2xd)*b1d;   Lxd2b1 = (b1xd)*b2d;
                I11dSim = DelW1d(1,n)*DelW1(1,n)/2; I22dSim = DelW2d(1,n)*DelW2(1,n)/2; I12dSim = DelW1d(1,n)*DelW2(1,n)/2; I21dSim = DelW2d(1,n)*DelW1(1,n)/2;
                MilSim(:,n+1) = Yn + a*Delt + b1*DelW1(1,n) + b2*DelW2(1,n) + Lx1b1*I11Sim + Lx1b2*I12Sim + Lx2b1*I21Sim + Lx2b2*I22Sim + Lxd1b1*I11dSim + Lxd1b2*I12dSim + Lxd2b1*I21dSim + Lxd2b2*I22dSim;
                
                % MEM:
                Yn = MEM(:,n); 
                    Yd = MEM(:,n-N/T);
                a = B0(Yd);
                b1 = B1(Yd);
                b2 = B2(Yd);
                expOmega1 = expm( A0hatDelt + A1*DelW1(1,n) + A2*DelW2(1,n) );
                MEM(:,n+1) = expOmega1*(Yn + (a - A1*b1 - A2*b2)*Delt + b1*DelW1(1,n) + b2*DelW2(1,n) );
                      
                % MMSim:
                Yn = MMSim(:,n);
                    Yd = MMSim(:,n-N/T);
                if n == 2*N/T
                    Yd2 = X0;
                else
                    Yd2 = MMSim(:,n-2*N/T);
                end 
                a = B0(Yd);
                b1 = B1(Yd);
                b2 = B2(Yd);
                I10Sim = DelW1(1,n)*Delt/2; I01Sim = DelW1(1,n)*Delt - I10Sim; I20Sim = DelW2(1,n)*Delt/2; I02Sim = DelW2(1,n)*Delt - I20Sim;
                expOmega2 = expm( A0hatDelt + A1*DelW1(1,n) + A2*DelW2(1,n) + 1/2*( LieA0A1*(I10Sim-I01Sim)+LieA0A2*(I20Sim-I02Sim)+LieA1A2*(I21Sim-I12Sim) ) );
                Hx1b1 = (0)*(A1*Yn+b1)-A1*b1; % This 0 is actually the Jacobian, d(b1)/dx.   j = 1, l = 1
                Hx1b2 = (0)*(A2*Yn+b2)-A2*b1; %  j = 1, l = 2
                Hx2b1 = (0)*(A1*Yn+b1)-A1*b2; %  j = 2, l = 1
                Hx2b2 = (0)*(A2*Yn+b2)-A2*b2; %  j = 2, l = 1
                b1d = B1(Yd2); b2d = B2(Yd2);
                b1xd = B1xd(Yd); b2xd = B2xd(Yd);
                Hxd1b1 = b1xd*(A1*Yd+b1d);  % j = 1; l = 1;        
                Hxd1b2 = b1xd*(A2*Yd+b2d);  % j = 1; l = 2;        
                Hxd2b1 = b2xd*(A1*Yd+b1d);  % j = 2; l = 1;
                Hxd2b2 = b2xd*(A2*Yd+b2d);  % j = 2; l = 2;
                MMSim(:,n+1) = expOmega2*(Yn + (a - A1*b1 - A2*b2)*Delt + b1*DelW1(1,n) + b2*DelW2(1,n) + Hx1b1*I11Sim + Hx1b2*I21Sim + Hx2b1*I12Sim + Hx2b2*I22Sim + Hxd1b1*I11dSim + Hxd2b1*I12dSim + Hxd1b2*I21dSim + Hxd2b2*I22dSim );
                 
                if useRefinedMilsteinschemes == 1
                    
                    % MilRef:
                    Yn = MilRef(:,n);
                        Yd = MilRef(:,n-N/T);
                    if n == 2*N/T
                        Yd2 = X0;
                    else
                        Yd2 = MilRef(:,n-2*N/T);
                    end  
                    a = A0*Yn + B0(Yd);
                    b1 = A1*Yn + B1(Yd);  b1x = A1;
                    b2 = A2*Yn + B2(Yd);  b2x = A2;
                    I11Ref = (DelW1(1,n)^2-Delt)/2; I22Ref = (DelW2(1,n)^2-Delt)/2;
                        % I12ref calculation:
                        I12Ref = dW1(1,n*step)*dW2(1,n*step)/2;
                        for j=1:step-1
                            I12Ref = I12Ref + dW1(1,n*step+j)*dW2(1,n*step+j)/2 + dW2(1,n*step+j)*(W1(1,n*step+j)-W1(1,n*step));
                        end
                        I21Ref = DelW1(1,n)*DelW2(1,n) - I12Ref;
                    Lx1b1 = (b1x)*b1;   Lx2b2 = (b2x)*b2;   Lx1b2 = (b2x)*b1;   Lx2b1 = (b1x)*b2;
                    b1xd = B1xd(Yd); b2xd = B2xd(Yd); b1d = A1*Yd + B1(Yd2); b2d = A2*Yd + B2(Yd2);
                    Lxd1b1 = (b1xd)*b1d; Lxd2b2 = (b2xd)*b2d;   Lxd1b2 = (b2xd)*b1d;   Lxd2b1 = (b1xd)*b2d;
                    % IijdRef calculations:
                            % I11dRef:
                                I11dRef = dW1d(1,n*step)*dW1(1,n*step)/2;
                                for j=1:step-1
                                    I11dRef = I11dRef + dW1d(1,n*step+j)*dW1(1,n*step+j)/2 + dW1(1,n*step+j)*(W1d(1,n*step+j)-W1d(1,n*step));
                                end
                            % I12dRef:
                                I12dRef = dW1d(1,n*step)*dW2(1,n*step)/2;
                                for j=1:step-1
                                    I12dRef = I12dRef + dW1d(1,n*step+j)*dW2(1,n*step+j)/2 + dW2(1,n*step+j)*(W1d(1,n*step+j)-W1d(1,n*step));
                                end
                            % I21dRef:
                                I21dRef = dW1(1,n*step)*dW2d(1,n*step)/2;
                                for j=1:step-1
                                    I21dRef = I21dRef + dW1(1,n*step+j)*dW2d(1,n*step+j)/2 + dW2d(1,n*step+j)*(W1(1,n*step+j)-W1(1,n*step));
                                end
                            % I22dRef:
                                I22dRef = dW2(1,n*step)*dW2d(1,n*step)/2;
                                for j=1:step-1
                                    I22dRef = I22dRef + dW2(1,n*step+j)*dW2d(1,n*step+j)/2 + dW2d(1,n*step+j)*(W2(1,n*step+j)-W2(1,n*step));
                                end
                    I21dRef = DelW1(1,n)*DelW2d(1,n)-I21dRef;
                    I22dRef = DelW2(1,n)*DelW2d(1,n)-I22dRef;    
                    MilRef(:,n+1) = Yn + a*Delt + b1*DelW1(1,n) + b2*DelW2(1,n) + Lx1b1*I11Ref + Lx2b1*I21Ref + Lx1b2*I12Ref + Lx2b2*I22Ref + Lxd1b1*I11dRef + Lxd1b2*I12dRef + Lxd2b1*I21dRef + Lxd2b2*I22dRef;

                    % MMRef:
                    Yn = MMRef(:,n);
                        Yd = MMRef(:,n-N/T);
                    if n == 2*N/T
                        Yd2 = X0;
                    else
                        Yd2 = MMRef(:,n-2*N/T);
                    end 
                    a = B0(Yd);
                    b1 = B1(Yd);
                    b2 = B2(Yd);
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
                    Hx1b1 = (0)*(A1*Yn+b1)-A1*b1; % This 0 is actually the Jacobian, d(b1)/dx.   j = 1, l = 1
                    Hx1b2 = (0)*(A2*Yn+b2)-A2*b1; %  j = 1, l = 2
                    Hx2b1 = (0)*(A1*Yn+b1)-A1*b2; %  j = 2, l = 1
                    Hx2b2 = (0)*(A2*Yn+b2)-A2*b2; %  j = 2, l = 1
                    b1d = B1(Yd2); b2d = B2(Yd2);
                    b1xd = B1xd(Yd); b2xd = B2xd(Yd);
                    Hxd1b1 = b1xd*(A1*Yd+b1d);  % j = 1; l = 1;        
                    Hxd1b2 = b1xd*(A2*Yd+b2d);  % j = 1; l = 2;        
                    Hxd2b1 = b2xd*(A1*Yd+b1d);  % j = 2; l = 1;
                    Hxd2b2 = b2xd*(A2*Yd+b2d);  % j = 2; l = 2;
                    MMRef(:,n+1) = expOmega2*(Yn + (a - A1*b1 - A2*b2)*Delt + b1*DelW1(1,n) + b2*DelW2(1,n) + Hx1b1*I11Ref + Hx2b1*I12Ref + Hx1b2*I21Ref + Hx2b2*I22Ref + Hxd1b1*I11dRef + Hxd2b1*I12dRef + Hxd1b2*I21dRef + Hxd2b2*I22dRef );

                end

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
                    Yn = EM(:,n);
                        Yd = EM(:,n-N/T); 
                    a = A0*Yn + B0(Yd);
                    b1 = A1*Yn + B1(Yd);
                    b2 = A2*Yn + B2(Yd);
                    EM(:,n+1) = Yn + a*Delt + b1*DelW1(1,n) + b2*DelW2(1,n);
                                    
                    % MilSim:
                    Yn = MilSim(:,n);
                        Yd = MilSim(:,n-N/T);
                        Yd2 = MilSim(:,n-2*N/T); 
                    a = A0*Yn + B0(Yd);
                    b1 = A1*Yn + B1(Yd); b1x = A1;
                    b2 = A2*Yn + B2(Yd); b2x = A2;
                    I11Sim = (DelW1(1,n)^2-Delt)/2; I22Sim = (DelW2(1,n)^2-Delt)/2; I12Sim = DelW1(1,n)*DelW2(1,n)/2; I21Sim = DelW1(1,n)*DelW2(1,n)/2;
                    Lx1b1 = (b1x)*b1;   Lx2b2 = (b2x)*b2;   Lx1b2 = (b2x)*b1;   Lx2b1 = (b1x)*b2;
                    b1xd = B1xd(Yd); b2xd = B2xd(Yd); b1d = A1*Yd + B1(Yd2); b2d = A2*Yd + B2(Yd2);
                    Lxd1b1 = (b1xd)*b1d; Lxd2b2 = (b2xd)*b2d;   Lxd1b2 = (b2xd)*b1d;   Lxd2b1 = (b1xd)*b2d;
                    I11dSim = DelW1d(1,n)*DelW1(1,n)/2; I22dSim = DelW2d(1,n)*DelW2(1,n)/2; I12dSim = DelW1d(1,n)*DelW2(1,n)/2; I21dSim = DelW2d(1,n)*DelW1(1,n)/2;
                    MilSim(:,n+1) = Yn + a*Delt + b1*DelW1(1,n) + b2*DelW2(1,n) + Lx1b1*I11Sim + Lx1b2*I12Sim + Lx2b1*I21Sim + Lx2b2*I22Sim + Lxd1b1*I11dSim + Lxd1b2*I12dSim + Lxd2b1*I21dSim + Lxd2b2*I22dSim;
                    
                    % MEM:
                    Yn = MEM(:,n); 
                        Yd = MEM(:,n-N/T);
                    a = B0(Yd);
                    b1 = B1(Yd);
                    b2 = B2(Yd);
                    expOmega1 = expm( A0hatDelt + A1*DelW1(1,n) + A2*DelW2(1,n) );
                    MEM(:,n+1) = expOmega1*(Yn + (a - A1*b1 - A2*b2)*Delt + b1*DelW1(1,n) + b2*DelW2(1,n) );
                    
                    % MMSim:
                    Yn = MMSim(:,n);
                        Yd = MMSim(:,n-N/T);
                        Yd2 = MMSim(:,n-2*N/T);
                    a = B0(Yd);
                    b1 = B1(Yd);
                    b2 = B2(Yd);
                    I10Sim = DelW1(1,n)*Delt/2; I01Sim = DelW1(1,n)*Delt - I10Sim; I20Sim = DelW2(1,n)*Delt/2; I02Sim = DelW2(1,n)*Delt - I20Sim;
                    expOmega2 = expm( A0hatDelt + A1*DelW1(1,n) + A2*DelW2(1,n) + 1/2*( LieA0A1*(I10Sim-I01Sim)+LieA0A2*(I20Sim-I02Sim)+LieA1A2*(I21Sim-I12Sim) ) );
                    Hx1b1 = (0)*(A1*Yn+b1)-A1*b1; % This 0 is actually the Jacobian, d(b1)/dx.   j = 1, l = 1
                    Hx1b2 = (0)*(A2*Yn+b2)-A2*b1; %  j = 1, l = 2
                    Hx2b1 = (0)*(A1*Yn+b1)-A1*b2; %  j = 2, l = 1
                    Hx2b2 = (0)*(A2*Yn+b2)-A2*b2; %  j = 2, l = 1
                    b1d = B1(Yd2); b2d = B2(Yd2);
                    b1xd = B1xd(Yd); b2xd = B2xd(Yd);
                    Hxd1b1 = b1xd*(A1*Yd+b1d);  % j = 1; l = 1;        
                    Hxd1b2 = b1xd*(A2*Yd+b2d);  % j = 1; l = 2;        
                    Hxd2b1 = b2xd*(A1*Yd+b1d);  % j = 2; l = 1;
                    Hxd2b2 = b2xd*(A2*Yd+b2d);  % j = 2; l = 2;
                    MMSim(:,n+1) = expOmega2*(Yn + (a - A1*b1 - A2*b2)*Delt + b1*DelW1(1,n) + b2*DelW2(1,n) + Hx1b1*I11Sim + Hx1b2*I21Sim + Hx2b1*I12Sim + Hx2b2*I22Sim + Hxd1b1*I11dSim + Hxd2b1*I12dSim + Hxd1b2*I21dSim + Hxd2b2*I22dSim );
    
                    if useRefinedMilsteinschemes == 1
                        
                        % MilRef:
                        Yn = MilRef(:,n);
                            Yd = MilRef(:,n-N/T);
                            Yd2 = MilRef(:,n-2*N/T);
                        a = A0*Yn + B0(Yd);
                        b1 = A1*Yn + B1(Yd);  b1x = A1;
                        b2 = A2*Yn + B2(Yd);  b2x = A2;
                        I11Ref = (DelW1(1,n)^2-Delt)/2; I22Ref = (DelW2(1,n)^2-Delt)/2;
                            % I12ref calculation:
                            I12Ref = dW1(1,n*step)*dW2(1,n*step)/2;
                            for j=1:step-1
                                I12Ref = I12Ref + dW1(1,n*step+j)*dW2(1,n*step+j)/2 + dW2(1,n*step+j)*(W1(1,n*step+j)-W1(1,n*step));
                            end
                            I21Ref = DelW1(1,n)*DelW2(1,n) - I12Ref;
                        Lx1b1 = (b1x)*b1;   Lx2b2 = (b2x)*b2;   Lx1b2 = (b2x)*b1;   Lx2b1 = (b1x)*b2;
                        b1xd = B1xd(Yd); b2xd = B2xd(Yd); b1d = A1*Yd + B1(Yd2); b2d = A2*Yd + B2(Yd2);
                        Lxd1b1 = (b1xd)*b1d; Lxd2b2 = (b2xd)*b2d;   Lxd1b2 = (b2xd)*b1d;   Lxd2b1 = (b1xd)*b2d;
                        % IijdRef calculations:
                                % I11dRef:
                                    I11dRef = dW1d(1,n*step)*dW1(1,n*step)/2;
                                    for j=1:step-1
                                        I11dRef = I11dRef + dW1d(1,n*step+j)*dW1(1,n*step+j)/2 + dW1(1,n*step+j)*(W1d(1,n*step+j)-W1d(1,n*step));
                                    end
                                % I12dRef:
                                    I12dRef = dW1d(1,n*step)*dW2(1,n*step)/2;
                                    for j=1:step-1
                                        I12dRef = I12dRef + dW1d(1,n*step+j)*dW2(1,n*step+j)/2 + dW2(1,n*step+j)*(W1d(1,n*step+j)-W1d(1,n*step));
                                    end
                                % I21dRef:
                                    I21dRef = dW1(1,n*step)*dW2d(1,n*step)/2;
                                    for j=1:step-1
                                        I21dRef = I21dRef + dW1(1,n*step+j)*dW2d(1,n*step+j)/2 + dW2d(1,n*step+j)*(W1(1,n*step+j)-W1(1,n*step));
                                    end
                                % I22dRef:
                                    I22dRef = dW2(1,n*step)*dW2d(1,n*step)/2;
                                    for j=1:step-1
                                        I22dRef = I22dRef + dW2(1,n*step+j)*dW2d(1,n*step+j)/2 + dW2d(1,n*step+j)*(W2(1,n*step+j)-W2(1,n*step));
                                    end
                        I21dRef = DelW1(1,n)*DelW2d(1,n)-I21dRef;
                        I22dRef = DelW2(1,n)*DelW2d(1,n)-I22dRef;    
                        MilRef(:,n+1) = Yn + a*Delt + b1*DelW1(1,n) + b2*DelW2(1,n) + Lx1b1*I11Ref + Lx2b1*I21Ref + Lx1b2*I12Ref + Lx2b2*I22Ref + Lxd1b1*I11dRef + Lxd1b2*I12dRef + Lxd2b1*I21dRef + Lxd2b2*I22dRef;
    
                        % MMRef:
                        Yn = MMRef(:,n);
                            Yd = MMRef(:,n-N/T);
                            Yd2 = MMRef(:,n-2*N/T);
                        a = B0(Yd);
                        b1 = B1(Yd);
                        b2 = B2(Yd);
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
                        Hx1b1 = (0)*(A1*Yn+b1)-A1*b1; % This 0 is actually the Jacobian, d(b1)/dx.   j = 1, l = 1
                        Hx1b2 = (0)*(A2*Yn+b2)-A2*b1; %  j = 1, l = 2
                        Hx2b1 = (0)*(A1*Yn+b1)-A1*b2; %  j = 2, l = 1
                        Hx2b2 = (0)*(A2*Yn+b2)-A2*b2; %  j = 2, l = 1
                        b1d = B1(Yd2); b2d = B2(Yd2);
                        b1xd = B1xd(Yd); b2xd = B2xd(Yd);
                        Hxd1b1 = b1xd*(A1*Yd+b1d);  % j = 1; l = 1;        
                        Hxd1b2 = b1xd*(A2*Yd+b2d);  % j = 1; l = 2;        
                        Hxd2b1 = b2xd*(A1*Yd+b1d);  % j = 2; l = 1;
                        Hxd2b2 = b2xd*(A2*Yd+b2d);  % j = 2; l = 2;
                        MMRef(:,n+1) = expOmega2*(Yn + (a - A1*b1 - A2*b2)*Delt + b1*DelW1(1,n) + b2*DelW2(1,n) + Hx1b1*I11Ref + Hx2b1*I12Ref + Hx1b2*I21Ref + Hx2b2*I22Ref + Hxd1b1*I11dRef + Hxd2b1*I12dRef + Hxd1b2*I21dRef + Hxd2b2*I22dRef );
    
                    end
    
                end
            end
        end
        % End for time (Ti-1)*tau <= t < Ti*tau, for Ti >= 4.

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

        for ti = 1:T
            EMvalues(:,ti,trial,h) = EM(:,ti*N/T);
            MilSimvalues(:,ti,trial,h) = MilSim(:,ti*N/T);
            MEMvalues(:,ti,trial,h) = MEM(:,ti*N/T);
            MMSimvalues(:,ti,trial,h) = MMSim(:,ti*N/T);
            if useRefinedMilsteinschemes == 1
                MilRefvalues(:,ti,trial,h) = MilRef(:,ti*N/T);
                MMRefvalues(:,ti,trial,h) = MMRef(:,ti*N/T);
            end
        end

        for ti = 1:T
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

for ti = 1:T
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

figure(T+1)
hold on
plot([-1,0,t],[X0(samplecomponent),X0(samplecomponent),sampleXref],'k')
plot([-1,0,sampletime],[X0(samplecomponent),X0(samplecomponent),sampleEM],'r')
plot([-1,0,sampletime],[X0(samplecomponent),X0(samplecomponent),sampleMilSim],'color',[1,1/2,0])
if useRefinedMilsteinschemes == 1
    plot([-1,0,sampletime],[X0(samplecomponent),X0(samplecomponent),sampleMilRef],'-square','color',[1,1/2,0])
end
plot([-1,0,sampletime],[X0(samplecomponent),X0(samplecomponent),sampleMEM],'b')
plot([-1,0,sampletime],[X0(samplecomponent),X0(samplecomponent),sampleMMSim],'color',[0,2/3,0])
if useRefinedMilsteinschemes == 1
    plot([-1,0,sampletime],[X0(samplecomponent),X0(samplecomponent),sampleMMRef],'-square','color',[0,2/3,0])
end
if useRefinedMilsteinschemes == 0
    legend(['Reference, $h = $ ',num2str(round(refdt,7))],['EM, $h = $ ',num2str(round(samplehvalue,3))],['Milstein (Simple), $h = $ ',num2str(round(samplehvalue,3))],['MEM, $h = $ ',num2str(round(samplehvalue,3))],['MM (Simple), $h = $ ',num2str(round(samplehvalue,3))],'Interpreter','latex','FontSize',12,'location','Northwest')
elseif useRefinedMilsteinschemes == 1
    legend(['Reference, $h = $ ',num2str(round(refdt,7))],['EM, $h = $ ',num2str(round(samplehvalue,3))],['Milstein (Simple), $h = $ ',num2str(round(samplehvalue,3))],['Milstein (Refined), $h = $ ',num2str(round(samplehvalue,3))],['MEM, $h = $ ',num2str(round(samplehvalue,3))],['MM (Simple), $h = $ ',num2str(round(samplehvalue,3))],['MM (Refined), $h = $ ',num2str(round(samplehvalue,3))],'Interpreter','latex','FontSize',12,'location','Northwest')
end
xlabel('$t$','Interpreter','latex','FontSize',12)
ylabel('$X_1(t)$','Interpreter','latex','FontSize',12)
title('Sample Trajectory')
axis([-1,T,min(sampleXref)-0.15,max(sampleXref)+0.15])
set(gcf,'position',[200,75,700,500])

%% ERROR GRAPHS:

% Calculate the bounds for the axes:
axisbounds = zeros(2,T);
for k = 1:2:T
    lowerk = k;
    if k < T
        upperk = k+1;
    else
        upperk = k;
    end
        axisbounds(1,k) = log10(min([MSErrorEM(lowerk,:),MSErrorMEM(lowerk,:),MSErrorMilSim(lowerk,:),MSErrorMMSim(lowerk,:)]));
        axisbounds(2,k) = log10(max([MSErrorEM(upperk,:),MSErrorMEM(upperk,:),MSErrorMilSim(upperk,:),MSErrorMMSim(upperk,:)]));
        if useRefinedMilsteinschemes == 1
            axisbounds(1,k) = log10(min([MSErrorEM(lowerk,:),MSErrorMEM(lowerk,:),MSErrorMilSim(lowerk,:),MSErrorMMSim(lowerk,:),MSErrorMilRef(lowerk,:),MSErrorMMRef(lowerk,:)]));
            axisbounds(2,k) = log10(max([MSErrorEM(upperk,:),MSErrorMEM(upperk,:),MSErrorMilSim(upperk,:),MSErrorMMSim(upperk,:),MSErrorMilRef(upperk,:),MSErrorMMRef(upperk,:)]));
        end
    if k < T
        axisbounds(1,k+1) = log10(min([MSErrorEM(lowerk,:),MSErrorMEM(lowerk,:),MSErrorMilSim(lowerk,:),MSErrorMMSim(lowerk,:)]));
        axisbounds(2,k+1) = log10(max([MSErrorEM(upperk,:),MSErrorMEM(upperk,:),MSErrorMilSim(upperk,:),MSErrorMMSim(upperk,:)]));
        if useRefinedMilsteinschemes == 1
            axisbounds(1,k+1) = log10(min([MSErrorEM(lowerk,:),MSErrorMEM(lowerk,:),MSErrorMilSim(lowerk,:),MSErrorMMSim(lowerk,:),MSErrorMilRef(lowerk,:),MSErrorMMRef(lowerk,:)]));
            axisbounds(2,k+1) = log10(max([MSErrorEM(upperk,:),MSErrorMEM(upperk,:),MSErrorMilSim(upperk,:),MSErrorMMSim(upperk,:),MSErrorMilRef(upperk,:),MSErrorMMRef(upperk,:)]));
        end
    end

end

% Error graph (t=Ti), for Ti = 1,2,...,T:
for Ti = T %1:T
    figure(Ti)
    hold on
    referenceorderhalf = max( [MSErrorEM(Ti,1),MSErrorMEM(Ti,1),MSErrorMilSim(Ti,1)] )*sqrt(hvalues); referenceorderone = min( [MSErrorEM(Ti,1),MSErrorMEM(Ti,1),MSErrorMilSim(Ti,1),MSErrorMMSim(Ti,1)] )*hvalues;
    plot(log(hvalues)/log(2),log10(referenceorderhalf)+0.2,'m-o','Linewidth',1)
    plot(log(hvalues)/log(2),log10(referenceorderone)-0.5,'m-square','Linewidth',1)
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
    axis([log(dt)/log(2),log(1)/log(2),axisbounds(1,Ti)-0.5,axisbounds(2,Ti)+0.25])
    ylabel('$\log_{10}\mathrm{MSE}(6)$','Interpreter','latex','FontSize',16)
    %title(['MS Strong Errors ','($t=$ ',num2str(Ti),')'],'Interpreter','latex','FontSize',16)
    set(gcf,'position',[200,75,700,500])
    grid on
    a = get(gca,'XTickLabel');
    set(gca,'XTickLabel',a,'fontsize',12)
    b = get(gca,'YTickLabel');
    set(gca,'YTickLabel',b,'fontsize',12)

        if Ti == 6 && runs == 5000
            saveas(gcf,'ZExample1.eps','epsc')
        end
end

disp(['Total simulation time is ',num2str(round(toc(simulationtime)/60,4)),' minutes.'])