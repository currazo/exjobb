%Minimum time optimal control of EECM model of Li-ion battery.

clc
clf
clear all
close all

import casadi.*
    
T = 32*60; % Time horizon
N = 50; % number of control intervals

%Ambient temperature
Tamb=20;
%Initial state of charge
z0=0.2;
%Requierd final state of charge
zf=0.95;

% SOC curve
%%% curve fitting
Psoc = [2285.40817707766,-10697.4574679954,21295.2972192728,-23497.8613791098,15701.5933529722,-6520.30071256223,1663.45936053633,-249.498983205122,20.4003983124595,2.50312504860982];


% R-RC-RC for charge:
p_R0_c = [-3.80682578389407e-10,4.00240273052387e-08,-1.58197429300026e-06,3.76270326950321e-05,-0.000858515247985248,0.0206147443533458];
p_R1_c = [0.0654615710184118,-0.00525650820371335,-0.253383096230716,0.000173883154055948,0.0254001255866514,0.438938254800170,-2.09393534667780e-06,-0.000862289009915430,-0.0367679565540722,0.0199638495732768,1.18335706702389e-05,0.000720704714780665,0.0312756100204243,-0.921502128136197,-1.39299889528213e-05,0.000405598247980637,-0.0311209477218485,0.906556369060044];
p_R2_c = [0.0234886933795704,-0.000425745554529780,-0.0618926972520154,-1.39712634099392e-05,0.00696626085724841,0.164437628770647,0.000125461481201322,-0.0439834373873193,0.776893090229784,-0.000283042412899104,0.0830810750719612,-2.41420032342403,0.000212163280209648,-0.0509303077624631,1.69286271232179];
p_C1_c = [403.383641284737,53.8385865437406,9648.20159841369,-157.992850174849,-25548.6534697763,220.088582815929,32490.6523045399,-33.6484580036925,-18272.1813194993];
p_C2_c = [-129159.217379799,-726.923924954005,2619841.98156477,16788.5803188600,-12858375.3146006,-37372.0814309199,26635079.1038597,40215.7527951502,-24824218.2071518,-18074.2244739902,8553853.51161107];

%Parameters related to the thermal model, assumed to be correct
Ccore=62.7; %J/K, Lumped Cell Core Heat Capacity
Cs=4.5; %J/K, Lumped Cell Surface (Casing) Heat Capacity
Ru = 15; %K/W, convection resistance; for natural convection, this number is above 10
Ru = 3.08; %K/W, convection resistance for forced convection
Rcore=1.94; %K/W, conduction resistance;
Q1=2.3;

% Declare model variables
v1 = SX.sym('v1');
v2 = SX.sym('v2');
z =  SX.sym('z');
Tcore = SX.sym('Tcore');
Ts = SX.sym('Ts');

%State vector and input current
x = [v1; v2; z; Tcore; Ts];
i = SX.sym('i');

%Define parameters
R0_ref = p_R0_c(6) + p_R0_c(5) * Tcore + p_R0_c(4) * Tcore^2 + p_R0_c(3) * Tcore^3 + p_R0_c(2) * Tcore^4 + p_R0_c(1) * Tcore^5;
R1_ref = p_R1_c(1) + p_R1_c(2)*Tcore + p_R1_c(3)*z + p_R1_c(4)*Tcore^2 + p_R1_c(5)*Tcore*z + p_R1_c(6)*z^2 + p_R1_c(7)*Tcore^3 + p_R1_c(8)*Tcore^2*z + p_R1_c(9)*Tcore*z^2 + p_R1_c(10)*z^3 + p_R1_c(11)*Tcore^3*z + p_R1_c(12)*Tcore^2*z^2 ...
+ p_R1_c(13)*Tcore*z^3 + p_R1_c(14)*z^4 + p_R1_c(15)*Tcore^3*z^2 + p_R1_c(16)*Tcore^2*z^3 + p_R1_c(17)*Tcore*z^4 + p_R1_c(18)*z^5;
R2_ref = p_R2_c(1) + p_R2_c(2)*Tcore + p_R2_c(3)*z + p_R2_c(4)*Tcore^2 + p_R2_c(5)*Tcore*z + p_R2_c(6)*z^2 + p_R2_c(7)*Tcore^2*z + p_R2_c(8)*Tcore*z^2 + p_R2_c(9)*z^3 + p_R2_c(10)*Tcore^2*z^2 ...
+ p_R2_c(11)*Tcore*z^3 + p_R2_c(12)*z^4 + p_R2_c(13)*Tcore^2*z^3 + p_R2_c(14)*Tcore*z^4 + p_R2_c(15)*z^5;
C1_ref = p_C1_c(1) + p_C1_c(2)*Tcore + p_C1_c(3)*z + p_C1_c(4)*Tcore*z + p_C1_c(5)*z^2 + p_C1_c(6)*Tcore*z^2 + p_C1_c(7)*z^3 + p_C1_c(8)*Tcore*z^3 + p_C1_c(9)*z^4;
C2_ref = p_C2_c(1) + p_C2_c(2)*Tcore + p_C2_c(3)*z + p_C2_c(4)*Tcore*z + p_C2_c(5)*z^2 + p_C2_c(6)*Tcore*z^2 + p_C2_c(7)*z^3 + p_C2_c(8)*Tcore*z^3 + p_C2_c(9)*z^4 + p_C2_c(10)*Tcore*z^4 + p_C2_c(11)*z^5;



%Define errors in perncetege for the parameters
tolerances={[0.8 1 1.2]
            [1]
            [1]
            [1]
            [1]};
%Generate all parameter combinatios
comb=cartesianProduct(tolerances);

%How many combinations there are
nrComb=size(comb,1);

% Define model dynamics and create the functions for them
f_set={};
for k=1:nrComb
    %Change values of the parameters and redefine Qb
    R0=R0_ref*comb(k,1);
    R1=R1_ref*comb(k,2);
    R2=R2_ref*comb(k,3);
    C1=C1_ref*comb(k,4);
    C2=C2_ref*comb(k,5);
    
    Qb=(R0*i^2);
    
    %Define dynamics
    xdot =   [-v1/R1/C1 + 1/C1*i;
              -v2/R2/C2 + 1/C2*i;
              -i/3600/Q1; 
              ((Ts-Tcore)/Rcore+Qb)/Ccore;
              ((Tamb-Ts)/Ru-(Ts-Tcore)/Rcore)/Cs];
    
    f_t = Function('f', {x, i}, {xdot});
    f_set=[f_set; {f_t}];
end

%Minimum time variables
xi=MX.sym('xi',1,nrComb);

% Fixed step Runge-Kutta 4 integrator for all combinations
F_set={};
for k=1:nrComb
    M = 4; % RK4 steps per interval
    DT = T/N/M;
    X0 = MX.sym('X0', 5);
    U = MX.sym('U');
    X = X0;
    Q = 0;
    f=f_set{k};
    for j=1:M
       k1 = f(X, U);
       k2 = f(X + DT/2 * k1, U);
       k3 = f(X + DT/2 * k2, U);
       k4 = f(X + DT * k3, U);
       X=X + xi(k)*DT/6*(k1 +2*k2 +2*k3 +k4);
    end

    %Integrator function
    F_t = Function('F', {X0, U}, {X}, {'x0','i'}, {'xf'});
    F_set=[F_set,{F_t}];
end

% Start with an empty NLP
w={};       %Symbolic set of states
w0 = [];    %Initial values of states
lbw = [];   %Lower bound for states
ubw = [];   %Upper bound for states
J = 0;      %Cost function
g={};        %Expressions with states and inputs which you can apply constraints to
lbg = [];  ubg = []; %Lower and upper bounds for g

%Decide x0
x0=[0; 0; z0; Tamb; Tamb];

for k=0:N-1
    % New NLP variable for the control
    Uk = MX.sym(['U_' num2str(k)]);

    w = {w{:}, Uk};
    %Input constraints
    lbw = [lbw; -30*Q1];
    ubw = [ubw;  5*Q1];
    w0 = [w0;  0];
end

for m=1:nrComb
    Xk = MX.sym(['X0_' num2str(m)], 5);
    w = {w{:}, Xk};
    
    lbw = [lbw; x0];
    ubw = [ubw; x0];
    w0 = [w0; x0];

    % Formulate the NLP
    for k=0:N-1

        % Integrate till the end of the interval
        F=F_set{m};
        Fk = F('x0', Xk, 'i', w{k+1});
        Xk_end = Fk.xf;

        % New NLP variable for state at end of interval
        Xk = MX.sym(['X_' num2str(k+1) '_' num2str(m)], 5);
        w = [w, {Xk}];
        %Bounds for Xk
        lbw = [lbw;  -0.06; -inf; 0.05; 5; 0];
        ubw = [ubw;  0.06;  inf; 0.95; 40; 50];
        w0 = [w0; 0; 0; 0; 0; 0];

        % Add equality constraint to make sure that it's continous and
        % follows the dynamics
        g = [g, {Xk_end-Xk}];
        lbg = [lbg; 0; 0; 0; 0; 0];
        ubg = [ubg; 0; 0; 0; 0; 0];
    end

    %Add constraints to NLP
    g=[g, w(end)];
    lbg = [lbg; -inf; -inf;  zf; -inf; -inf];
    ubg = [ubg;  inf;  inf;  inf;  40; inf];


    
end

% Cost function as xi only
J=xi*ones(nrComb,1);

%With an initial value of 1
w0=[w0; ones(nrComb,1)];
%Upper and lower bound for xi
lbw = [lbw;  0.5*ones(nrComb,1)];
ubw = [ubw;  2*ones(nrComb,1)];
    
%  Create an NLP solver and add xi as a variable
prob = struct('f', J, 'x', [vertcat(w{:}); xi'], 'g', vertcat(g{:}));
solver = nlpsol('solver', 'ipopt', prob);

% Solve the NLP
sol = solver('x0', w0, 'lbx', lbw, 'ubx', ubw,...
            'lbg', lbg, 'ubg', ubg);
w_opt = full(sol.x);

%Extract states and inputs from the optimal solution
x1_opt = w_opt((N+1):5:(end-nrComb));
x2_opt = w_opt((N+2):5:(end-nrComb));
x3_opt = w_opt((N+3):5:(end-nrComb));
x4_opt = w_opt((N+4):5:(end-nrComb));
x5_opt = w_opt((N+5):5:(end-nrComb));
u_opt =  w_opt(1:N);
xi_opt=w_opt((end-nrComb+1):end);

%Time
tgrid = linspace(0, T, N+1);



%Minutes to fully charge
ChargeTime=T*xi_opt/60

%Plot results
for k=0:(nrComb-1)
    int=N*k + (k+1):(k+1)+N*(k+1);
    t=tgrid*xi_opt(k+1);
    subplot(2,2,1)
    hold on
    plot(t, x1_opt(int), '-')
    title('v1')
    subplot(2,2,2)
    hold on
    plot(t, x3_opt(int), '-')
    title('z')
    subplot(2,2,3)
    hold on
    plot(t, x4_opt(int), '-')
    title('Tcore')
end

subplot(2,2,4)
plot(linspace(0, T, N), -u_opt, '-')
title('i')
ylim([0 ceil(5*4.4+1)])