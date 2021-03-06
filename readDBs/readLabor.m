% Exemplo de chamada:
% dataFull = readLabor('data/labor/labor-neg.data.txt');
% data = dataFull{2};
% labels = dataFull{1};
%
% $Author: Derzu Omaia
function [data] = readLabor(path)
    file = fopen(path);
    if file<=0
        fprintf('Error file \"%s\" not found\n', path);
        return;
    end
    
    data = textscan(file,'%f %f %f %f %s %f %s %f %f %s %f %s %s %s %s %s %s','Delimiter',',','CollectOutput',1,'EmptyValue',0, 'TreatAsEmpty', '?');
    fclose(file);
    %data
end