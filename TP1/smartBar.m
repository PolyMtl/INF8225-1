function [ output_args ] = smartBar( probaJointe, events, names )

    if length(probaJointe) ~= length(events)
       warning('probaJointe and events should have the same length')
    end

    XLegend = {};
    for i=1:length(events)
        e=events(i,:);
        legend = '';
        for j=1:length(names)
            if e(j)==2
                legend = [legend ' !' names(j)];
            else
                legend = [legend '  ' names(j)];
            end
        end
        XLegend{1,i} = legend;
    end
    
    figure();
    bar([1:length(probaJointe)], probaJointe,1);
    xticklabelRot(XLegend, 90, 10, 0.01);
    
end

