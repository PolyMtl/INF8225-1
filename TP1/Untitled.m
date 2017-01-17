C = 1; S = 2; R = 3; W = 4;
nvars = 4;

dgmJ = mkSprinklerDgm('infEngine', 'jtree');
dgmV = mkSprinklerDgm('infEngine', 'varelim');
dgmE = mkSprinklerDgm('infEngine', 'enum');

if ~isOctave
    drawNetwork('-adjMatrix', dgmJ.G, '-nodeLabels', {'C', 'S', 'R', 'W'},...
        '-layout', Treelayout);
end

pWj = dgmInferQuery(dgmJ, S);
pWv = dgmInferQuery(dgmV, W);
pWe = dgmInferQuery(dgmE, W);