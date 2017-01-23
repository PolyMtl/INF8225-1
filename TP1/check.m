%% Init
clear all;
close all;
clc;

% C = 1; S = 2; R = 3; W = 4; 
% 
% names = ['c' 's' 'r' 'w'];
% VRAI = 1; FAUX = 2;
% 
% pCond{C} = reshape([0.5 0.5], 2, 1);
% pCond{S} = reshape([0.5 0.9 0.5 0.1], 2, 2);
% pCond{R} = reshape([0.8 0.2 0.2 0.8], 2, 2);
% pCond{W} = reshape([1 0.1 0.1 0.01 0 0.9 0.9 0.99], 2, 2, 2);
% 
% pJointe = @(e) ( pCond{C}(e(C)) ...
%                * pCond{S}(e(C),e(S)) ...
%                * pCond{R}(e(C),e(R)) ...
%                * pCond{W}(e(S),e(R),e(W)));

A = 1; B = 2; C = 3;
names = ['a' 'b' 'c'];
FAUX = 1; VRAI = 2;

pCond{A} = reshape([0.6 0.4], 2, 1);
pCond{B} = reshape([0.9 0.1], 2, 1);
pCond{C} = reshape([0.5, 0.5, 0.4, 0.2, 0.5, 0.5, 0.6, 0.8],2,2,2);

 pJointe = @(e) ( pCond{A}(e(A)) ...
                * pCond{B}(e(B)) ...
                * pCond{C}(e(A),e(B),e(C)) );

%% Proba Jointe                   
[probaJointe Omega] = computeProba(pJointe, [0 0 0]);


%% Histogramme
smartBar(probaJointe, Omega, names);

