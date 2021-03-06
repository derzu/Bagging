% Gera T subconjutos de dados
%
% @param dados matriz MxD (linha x coluna). M amostras (instances), D atributos (features).
% @param labels vetor com os rotulos dos dados da matriz dados.
% @param T quantidade de classificadores a serem gerados no pool.
% @param m quantidade de amostras utilizadas para gerar cada classificador do pool. Opcional,
% caso seja omitido eh gerado um subconjunto com a mesma quantida de
% amostras de dados.
%
% @return o pool de T classificadores
%
% Forma de uso:
% bag = baggingPool(data, labels, 10, 'k');
%
% $Author: Derzu Omaia
function [pool] = baggingPool(dados, labels, T, varargin) % varargin eh uma lista de parametros.
    % trata se tem o parametro do tamanho do subconjunto, se nao tiver usa o default.
    if (length(varargin)==1)
        bag = bagging(dados, T, varargin{1});
    else
        bag = bagging(dados, T);
    end

    pool = cell(T, 1);
    
    % gera os dados provindos do bagging.
    for i=1:T
        data  = dados(bag(i, :), :);
        label = labels(bag(i,:));
        pool{i} = geraDecisionTree(data, label, 0);
    end
end