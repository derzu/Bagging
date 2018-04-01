% Classify the dataset using just ONE classifier.
%
% $Author: Derzu Omaia
function [ erros ] = classificaUm( c , data, labels)
    M = size(data,1); % M amostras no banco de teste

    if isa(c, 'prmapping')
        resultados = labeld(data, c);
    else
        resultados = predict(c,data);
    end
    
    %acertos = 0;
    erros = 0;
    for i=1:M
        if resultados(i)~=labels(i)
            erros = erros + 1;
        %else
        %    acertos = acertos + 1;
        end 
    end
    erros = erros/M;
    %acertos = acertos/M;
end

