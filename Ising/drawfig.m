%% binomial distribution
load('errors.mat');
load('errors2.mat');
fig = figure();
set (gcf,'Position',[400,100,700,600]);
x=linspace(1,7,7);
plot(x, errors2, 'r-s', 'linewidth',3,'markersize',18)
hold on
plot(x, errors,'-o','Color', [0 0.5 0],'linewidth',3,'markersize',18)
% grid on;
legend({'Exact Monte Carlo','Gradient Free SVGD'},'FontSize',20)
ylabel('log10 MSE')
xlabel('#Samples')
xlim([1,7])
set(gca,'FontSize',30);
set(gca, 'XTick', [1 4 7])
set(gca,'XTickLabel',{'1','10','100'})
fig1 = gcf;
fig1.PaperPositionMode = 'auto';
fig_pos = fig1.PaperPosition;
fig1.PaperSize = [fig_pos(3) fig_pos(4)];
print(fig,'binomial','-dpdf')

%% Ising model
load('ising.mat');
load('ising2.mat');
load('ising3.mat');
fig = figure();
set (gcf,'Position',[400,100,700,600]);
x=linspace(1,7,7);
plot(x, ising2, 'r-s', 'linewidth',3,'markersize',18)
hold on
plot(x, ising3, 'b-d', 'linewidth',3,'markersize',18)
plot(x, ising,'-o','Color', [0 0.5 0],'linewidth',3,'markersize',18)
% grid on;
legend({'Exact Monte Carlo','Gibbs Sampling','Gradient Free SVGD'},'FontSize',20)
ylabel('log10 MSE')
xlabel('#Samples')
xlim([1,7])
set(gca,'FontSize',30);
set(gca, 'XTick', [1 4 7])
set(gca,'XTickLabel',{'1','10','100'})
fig1 = gcf;
fig1.PaperPositionMode = 'auto';
fig_pos = fig1.PaperPosition;
fig1.PaperSize = [fig_pos(3) fig_pos(4)];
print(fig,'ising','-dpdf')

%% Discrete MRF
load('dmrferrors.mat');
load('dmrferrors2.mat');
fig = figure();
set (gcf,'Position',[400,100,700,600]);
x=linspace(1,7,7);
plot(x, dmrferrors2, 'b-d', 'linewidth',3,'markersize',18)
hold on
plot(x, dmrferrors,'-o','Color', [0 0.5 0],'linewidth',3,'markersize',18)
% grid on;
legend({'Gibbs Sampling','Gradient Free SVGD'},'FontSize',20)
ylabel('log10 MSE')
xlabel('#Samples')
xlim([1,7])
set(gca,'FontSize',30);
set(gca, 'XTick', [1 4 7])
set(gca,'XTickLabel',{'1','10','100'})
fig1 = gcf;
fig1.PaperPositionMode = 'auto';
fig_pos = fig1.PaperPosition;
fig1.PaperSize = [fig_pos(3) fig_pos(4)];
print(fig,'dMRF','-dpdf')

%% Ising model pairwise strength
load('ising_ps.mat');
load('ising2_ps.mat');
load('ising3_ps.mat');
fig = figure();
set (gcf,'Position',[400,100,700,600]);
x=linspace(-0.15,0.15,31);
plot(x, ising2_ps, 'r-', 'linewidth',3,'markersize',18)
hold on
plot(x, ising3_ps, 'b-', 'linewidth',3,'markersize',18)
plot(x, ising_ps,'-','Color', [0 0.5 0],'linewidth',3,'markersize',18)
% grid on;
legend({'Exact Monte Carlo','Gibbs Sampling','Gradient Free SVGD'},'Location','northwest','FontSize',20)
ylabel('log10 MSE')
xlabel('Pairwise Strength \sigma_p')
xlim([-0.15,0.15])
set(gca,'FontSize',30);
fig1 = gcf;
fig1.PaperPositionMode = 'auto';
fig_pos = fig1.PaperPosition;
fig1.PaperSize = [fig_pos(3) fig_pos(4)];
print(fig,'strength','-dpdf')

