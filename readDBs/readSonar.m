% Exemplo de chamada:
% [dadosX, dadosY] = readSonar('data/UCI/sonar/sonar.all-data.txt');
%
% $Author: Derzu Omaia
function [dataX, dataY] = readSonar(path)
    file = fopen(path);
    if file<=0
        fprintf('Error file \"%s\" not found\n', path);
        return;
    end

    data = textscan(file,'%f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %s', ...
                         'Delimiter',',','CollectOutput',1);
%     data = textscan(file,'%f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f', ...
%                          '%f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f', ...
%                          '%f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %s', ...
%                          'Delimiter',',','CollectOutput',1);
    fclose(file);
    %data
    
    dataX = data{1}; % info
    dataY = data{2}; % labels
    [~, ~, dataY] = unique(dataY); % converte os labels strings para ints
end