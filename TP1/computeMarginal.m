function [ proba ] = computeMarginal( probaJointe, event, knowing )
%COMPUTEMARGINAL Summary of this function goes here
%   Detailed explanation goes here

[p e] = computeProba( probaJointe, knowing + event);
p1 = sum(p);

[p e] = computeProba( probaJointe, knowing - event);
p2 = sum(p);

pEvent = p1;
pKnowing = p1+p2;

proba = pEvent/pKnowing;

end

