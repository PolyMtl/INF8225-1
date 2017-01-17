%%
clear all;
close all;

%%

B=1; F=2; G=3; D=4; FT=5;
names = cell(1,5);
names{B} = 'Battery';
names{F} = 'Fuel';
names{G} = 'Gauge';
names{D} = 'Distance';
names{FT} = 'FillTank';

%   B   F
%    \ /
%     v
%     G
%    / \
%   v   v
%   D   FT

dgm = zeros(5,5);
dgm([B F] ,G) = 1;
dgm(G,[D FT]) = 1;

drawNetwork('-adjMatrix', dgm, '-nodeLabels', {'B', 'F', 'G', 'D', 'FT'},...
        '-layout', Treelayout);


CPDs{B}  = tabularCpdCreate(reshape([0.1 0.9],2,1)); % B
CPDs{F}  = tabularCpdCreate(reshape([0.1 0.9],2,1)); % F
CPDs{G}  = tabularCpdCreate(reshape([0.9 0.8 0.8 0.2 0.1 0.2 0.2 0.8],2,2,2)); % B F G
CPDs{D}  = tabularCpdCreate(reshape([0.95 0.7 0.05 0.3],2,2)); % G D
CPDs{FT} = tabularCpdCreate(reshape([0.2 0.8 0.6 0.4],2,2));   % G FT 



dgm = dgmCreate(dgm, CPDs, 'nodenames', names, 'infEngine', 'jtree');


%%
pGBF = dgmInferQuery(dgm, [G B F]);