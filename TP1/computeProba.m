function [ proba, events ] = computeProba( probaJointe, eventSpace, distributionsSize )
%COMPUTEPROBA Summary of this function goes here
%   Detailed explanation goes here

    if nargin < 3, distributionsSize = ones(size(eventSpace))*2; end
    
    %Find index of event which are 'free' (which should take all their
    %possible values)
    freeP = [];
    forbiddenValues = [];
    for i=1:length(eventSpace)
        if eventSpace(i)<=0
            freeP = [freeP i];
            forbiddenValues = [forbiddenValues -eventSpace(i)];
        end
    end
    
    %% Init events Calculations
    distributionsSize = distributionsSize(freeP);
    
    function [value] = startValue(eventID)
        value = eventSpace(eventID);
        if value <= 0
            if value == -1
                value = 2;
            else
                value = 1;
            end
        end
    end
    
    function [newEvent] = incrementEvent(lastEvent)
        newEvent = lastEvent;
        freeIterator = 1;
        last = length(freeP)+1;
        while freeIterator < last
            v = 1 + newEvent(freeP(freeIterator));
            if v == forbiddenValues(freeIterator)
                v = v+1;
            end
            if v > distributionsSize(freeIterator)
                newEvent(freeP(freeIterator)) = startValue(freeP(freeIterator));
                freeIterator = freeIterator+1;
            else
                newEvent(freeP(freeIterator)) = v;
                freeIterator = last;
            end
        end
    end

    event = zeros(size(eventSpace));
    for i=1:length(event)
       event(i) = startValue(i); 
    end
    
    %% Compute probabilities
    nbrOfProbaToCompute = prod(distributionsSize-(forbiddenValues~=0));
    events = zeros(nbrOfProbaToCompute, length(eventSpace));
    proba = zeros(nbrOfProbaToCompute,1);
    for i=1:nbrOfProbaToCompute
       events(i,:) = event(:);
       proba(i)  = probaJointe(event);
       event = incrementEvent(event);
    end
    
end

