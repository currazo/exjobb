%Minimum time optimal control of EECM model of Li-ion battery...

clc
clf
clear all
close all

import casadi.*
    
T = 17*60; % Time horizon
N = 25; % number of control intervals

%Ambient temperature
Tamb=20;
%Initial state of charge
z0=0.2;
%Requierd final state of charge
zf=0.95;


%Declare model parameters
C1_ref = 1500;
Q1_ref = 4.4;

%Parameters related to the thermal model, assumed to be correct
Ccore=148;
Cs=18.8;
Rcore=2;
Ru=0.65;

%Parameters for the equations for Re & Ro
a2=4e-3;
a4=8e-3;
b2=4.2e-5;
b4=2.5e-5;
gamma=3.6e-3;

% Declare model variables
v1 = SX.sym('v1');
dz = SX.sym('dz');
z =  SX.sym('z');
Tcore = SX.sym('Tcore');
Ts = SX.sym('Ts');

%State vector and input current
x = [v1; dz; z; Tcore; Ts];
i = SX.sym('i');

%Resistance models
Re_ref=a2-b2*Tcore;
Ro_ref=a4-b4*Tcore-gamma*z;

%Define errors in perncetege for the parameters
tolerances={[0.8 1 1.2],
            [1],
            [1],
            [1]};
%Generate all parameter combinatios
comb=cartesianProduct(tolerances);

%How many combinations there are
nrComb=size(comb,1);

% Define model dynamics and create the functions for them
f_set={};
for k=1:nrComb
    %Change values of the parameters and redefine Qb
    Re=Re_ref*comb(k,1);
    Ro=Ro_ref*comb(k,2);
    C1=C1_ref*comb(k,3);
    Q1=Q1_ref*comb(k,4);
    
    Qb=Re*i^2+v1^2/Ro;
    
    %Define dynamics
    xdot = [-v1/Ro/C1 + 1/C1*i;   
              -dz/Ro/C1 + 1/C1*i; 
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
    lbw = [lbw; -5*Q1];
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
        ubw = [ubw;  0.06;  inf; 0.95; 28; 50];
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
ylim([0 ceil(5*Q1+1)])
