clear% =========================================================================
% ERC Design Step 1: Spring Selection based on target torque and Safety
% Factor (SF)
%
% Author: Rui Wu (rui.wu@usys.ethz.ch)
%         Stefano Mintchev (stefano.mintchev@usys.ethz.ch)
% Environmental Robotics Lab, ETH Zurich, 2025
%
% Funded by the Horizon Europe project in AI & robotics:
% "SPEAR: Spatial Perception & Embodied Autonomy Research"
% =========================================================================

%% Input 1/2: Design parameters
SF = 1.5; % Target Safety Factor (SF), default = 2
n_spring = 1; % Number of springs
sensorised = 0; % Does the ERC has integrated angular sensor (potentiometer)? 1->Y, 0->N
L_coupler = 3; % Length of coupler for senrorised ERC
dtheta = 0.1*pi/180; % Resolution of cam design, default = 0.1*pi/180

%% Input 2/2: Required Response
% Need: 1) Theta_EA, Rotation of one cam (1/2 of total bending), in radians
%       2) M_EA, Target torque profile (N*m)

% % Analytical formulation
% a = 0.12;
% b = 0.5*180/pi;
% c = 0.0;
% d = 0.035*a; % don't vary
% xlim = 77.92/180*pi;
% x = 0:0.001:xlim;
% f =  d*tan((pi*x)/(2*xlim)) + a * (c + (x.^(1/4)*exp(1/8).*exp(-b*x.^2))/(2^(1/2)/(4*b^(1/2)))^(1/4))/(c + 1);
% % % anti-symmetric
% % Theta_EA=[-flip(x(1:end-10)) x(2:end-10)]/2;
% % M_EA=[-flip(f(1:end-10)) f(2:end-10)];
% % asymmetric
% Theta_EA=[-0.25/180*pi x(1:end-10)]/2;
% M_EA=[-0.4 f(1:end-10)];
% 
% % 
% % loop comparison
% figure()
% hold on
% for i = 0.5:0.5:2.5
%     % Analytical formulation
%     a = 0.2;
%     b = i*180/pi;
%     c = 0.0;
%     d = 0.01*a; % don't vary
%     xlim = 77.92/180*pi;
%     x = 0:0.001:xlim;
%     f =  d*tan((pi*x)/(2*xlim)) + a * (c + (x.^(1/4)*exp(1/8).*exp(-b*x.^2))/(2^(1/2)/(4*b^(1/2)))^(1/4))/(c + 1);
%     % % anti-symmetric
%     % Theta_EA=[-flip(x(1:end-10)) x(2:end-10)]/2;
%     % M_EA=[-flip(f(1:end-10)) f(2:end-10)];
%     % asymmetric
%     Theta_EA=[-0.25/180*pi x(1:end-10)]/2;
%     M_EA=[-0.45 f(1:end-10)];
% 
%     plot(Theta_EA*2,M_EA)
% end

% left arm
threshold = 0.1;
slope = 1.;
% theta_lim = [-6 77.92];
theta_lim = [-6 77.92];
Theta_EA=[-2.5*threshold/slope threshold/slope 30 75.92 theta_lim(2)]/2*pi/180;
% M_EA=-[1.0 threshold -threshold 0 0 -1.0];
M_EA=-[2.5*threshold -threshold 0 0 -2.5*threshold];

% plot(Theta_EA*2,M_EA)
% hold off

% Previous
% endstop_angle = pi/3;
% angle_range = [-1 1]*(endstop_angle - 1.5e-2);
% a = 0.2;
% b = 300;
% Theta_EA = linspace(angle_range(1),angle_range(2), 1e3);
% M_EA = -(-a*(1./(1+exp(-b*Theta_EA)) - 0.5 + 0.1*Theta_EA + 2.5e-2*tan(Theta_EA/endstop_angle*pi/2)));


%%% Example 1
% Theta_EA=[-45 45]*pi/180;
% M_EA=[-1 1]*0.25;

% %%% Example 2
% Theta_EA=[-45 -44.9 -0.1 0.1 44.9 45]*pi/180;
% M_EA=[-1 1 -1 1 -1 1]*0.1;

% %%% Example 3
% Theta_EA=[-45 -0.1 0.1 45]*pi/180;
% M_EA=[-0.5 -0.5 0.5 0.5]*0.25;

% %%% Example 4
% Theta_EA=[-45 -0.1 0.1 45]*pi/180;
% M_EA=[-0 -0.5 0.5 0]*0.25;

%%% Example on repository
% % Rotation of one cam (1/2 of total bending), in radians
% Theta_EA = [-45 -44.9 -30 -15 0 0.1 15 30 44.9 45] * pi / 180;  
% % Target torque profile (N*m)
% M_EA = [-0.4 -0.035 0 -0.07 -0.035 0.035 0.07 0 0.035 0.4];  

% %%% Live demo
% M_max=0.05; % Target maximum moment (N*m)
% Theta_EA=(-15:0.25:15);
% M_EA=(15^2-Theta_EA.^2).^0.5/30+0.5; % torque (N*m)
% Theta_EA=[0.1 10 Theta_EA+30]*pi/180; % rotation of one rolamite (1/2 of total bending)
% M_EA=[1 1 M_EA]*M_max; % normalised moment profile * M_max
% M_EA(end)=5*M_max; % adding an end-stopping effect
% M_EA=[fliplr(-M_EA) M_EA]; % generate symmetrical profile
% Theta_EA=[fliplr(-Theta_EA) Theta_EA]; % generate symmetrical profile

% %%% SquAshy 2025
% % Rotation of one cam (1/2 of total bending), in radians
% Theta_EA = [45 45.2 50 55:1:90 90.2]/2 * pi / 180;  
% % Target torque profile (N*m)
% M_EA = [-0.5 0.023 0.023 0.05*0.2*sin((55:1:90)*pi/180) 0.5]; 

%%% X-drone 2025
% Rotation of one cam (1/2 of total bending), in radians
% Theta_EA = [45 45.1 50 75 79.9 80] * pi / 180;  
% Target torque profile (N*m)
% M_EA = [-0.6 0.18 0.18 0.08 0.08 0.8]; 

%% Spring Requirement Evaluation

% Interpolate the target torque according to resolution
theta = (Theta_EA(1):dtheta:Theta_EA(end)); % acquisition points
M = interp1(Theta_EA, M_EA, theta, 'linear'); % target torque (N*m)
DU = cumtrapz(theta*2, M);% incremental energy starting from theta = 0 (J)
% Target torque variation rate
slopes = diff(M_EA) ./ diff(Theta_EA * 2);  
% Maximum target torque reduction rate
M_slp = -1 * min(slopes);  

%% Output
close all

D = max((0.28*M_slp/n_spring*1000)^(1/3),(0.33*(max(DU) - min(DU))/n_spring*1000)^(1/3)); % Mass model's estimation of spring diameter
fprintf('Spring requirement:\n');
fprintf('   No. of springs (user defined):  %g \n', n_spring);
fprintf('   Max length & tension:           L_max*T_max >= %.2g N*m (per spring)\n', 4 * M_slp/n_spring * SF);
fprintf('   Max. elongation & tension:     ΔL_max*T_max >= %.2g N*m (per spring)\n\n', 2 * (max(DU) - min(DU))/n_spring * SF);
fprintf('Mass model estimation (this is only a guideline, as spring selection is not unique):\n');
fprintf('L_max: %.2g mm, coil diameter: %.2g mm, wire diameter: %.2g mm, ', D*10 + sensorised*L_coupler, D, D/5);
fprintf('ERC mass: %.2g g\n', max(max(0.14 * M_slp/n_spring*1000, 0.17 * (max(DU) - min(DU))/n_spring*1000), 25)+sensorised*10);
if sensorised == 1
    fprintf('Note: L_max includes the length of coupler, and ERC mass includes 10 g of sensor accesories\n\n');
else
    fprintf('\n');
end
fprintf('For max. torque reduction rate %.3g N*m/rad\n', M_slp);
fprintf('and target energy variation %.3g J\n', (max(DU) - min(DU)));
fprintf('with Safety Factor SF = %.3g \n', SF);

figure
plot(Theta_EA*2*180/pi,M_EA,'LineWidth', 1.5)
xlabel('Rotation angle [degree]'); ylabel('Torque [Nm]')
fontsize(16,"points")
title('Target ERC torque response')
grid on