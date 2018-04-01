%EXIBEGRAFICOS Summary of this function goes here
%
% $Author: Derzu Omaia
function displayChartSingle(acc_m, Lini, Lp, Lfim, leg1, leg2)
linhas = [{'--b*'}; {'-ro'}; {':g+'}; {'--r*'}; {'-go'}; {':b+'}; {'--g*'}; {'-bo'}; {':r+'}];

% Exibicao do grafico
figure();
x = [Lini:Lp:Lfim];

plot(x, acc_m, linhas{2}, 'LineWidth', 2);

plot(x, acc_m(2, :), linhas{2}, 'LineWidth', 2);
hold on;
plot(x, acc_m(1, :), linhas{1}, 'LineWidth', 2);

title('Bagging');
xlabel('L');
ylabel('Accurace Rate');
legend(leg1, leg2);
end

