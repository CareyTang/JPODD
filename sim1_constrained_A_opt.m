clear;clc;close all;

mc_trials = 100;
seed0 = 55;

M = 8;
sigma2 = deg2rad(5)^2;
eps_s = 1e-2;
reg_eps = 1e-8;

maxIter = 100;
stop_rel = 1e-6;

big_center = [0; 0; 0];
big_size = [500; 500; 300];
target_offset_z = 200;
target_size = [20; 20; 20];

vertex_box_center = big_center;
vertex_cube_size = [500; 500; 300];
vertex_box_size = [60; 60; 60];

beta_max_deg = 90;
tilt_constraint_tol = 1e-10;

eta0 = 1.0;
beta_bt = 2.0;
fd_eps = 1e-6;
initMaxTry = 80;
verbose_outer = false;

%% Method selection
method_ids = {'gs_sqp', 'iso_sqp', 'manopt', 'tr_sqp', 'pso', 'de'};
method_names = {'GS-SQP', 'Iso-SQP', 'Manopt', 'TR-SQP', 'PSO', 'DE'};
n_methods = numel(method_ids);

% GS-SQP.
gs_sqp = struct();
gs_sqp.etaR0 = max(eta0, 1e-6);
gs_sqp.etaT0 = max(eta0, 1e-6);
gs_sqp.secantEps = 1e-10;
gs_sqp.minDirectionalCurvature = 1e-8;
gs_sqp.minModelEig = 1e-8;
gs_sqp.tau0 = 1e-8;
gs_sqp.tauScale = 10;
gs_sqp.tauMax = 1e8;
gs_sqp.qpMaxAttempts = 6;
gs_sqp.gradTol = 1e-10;
gs_sqp.stepTol = 1e-10;
gs_sqp.acceptTol = 1e-10;
gs_sqp.qpAlgorithm = 'active-set';
gs_sqp.qpMaxIterations = 20;
gs_sqp.qpOptimalityTolerance = 1e-6;
gs_sqp.qpConstraintTolerance = 1e-8;
gs_sqp.qpUseWarmStart = true;
gs_sqp.quadprog = struct();

% Iso-SQP.
generic_mm = struct();
generic_mm.L0 = max(eta0, 1e-6);
generic_mm.LScale = max(beta_bt, 1.1);
generic_mm.LMax = 1e8;
generic_mm.qpMaxAttempts = 25;
generic_mm.stepTol = 1e-10;
generic_mm.acceptTol = 1e-10;
generic_mm.majorizeTol = 1e-10;
generic_mm.qpAlgorithm = 'active-set';
generic_mm.qpMaxIterations = 20;
generic_mm.qpOptimalityTolerance = 1e-6;
generic_mm.qpConstraintTolerance = 1e-8;
generic_mm.qpUseWarmStart = true;
generic_mm.quadprog = struct();

% Manopt.
manopt = struct();
manopt.maxIter = 50;
manopt.tolGradNorm = 1e-6;
manopt.fdEps = 1e-5;

% TR-SQP.
tr_sqp = struct();
tr_sqp.maxIter = 80;
tr_sqp.innerMaxIter = 20;
tr_sqp.innerMaxFunctionEvaluations = 800;
tr_sqp.delta0 = 60;
tr_sqp.deltaMin = 1e-5;
tr_sqp.deltaMax = 300;
tr_sqp.deltaShrink = 0.5;
tr_sqp.deltaGrow = 1.8;
tr_sqp.eta2 = 0.75;
tr_sqp.acceptTol = 1e-10;
tr_sqp.stepTol = 1e-8;
tr_sqp.optimalityTol = 1e-6;
tr_sqp.constraintTol = 1e-8;

% PSO.
pso = struct();
pso.swarmSize = 40;
pso.maxIter = 80;
pso.inertia = 0.72;
pso.cognitive = 1.49;
pso.social = 1.49;

% DE.
de = struct();
de.populationSize = 50;
de.maxIter = 80;
de.mutationFactor = 0.8;
de.crossoverRate = 0.9;

%% Monte Carlo simulation
fprintf('Running constrained A-opt with %d Monte Carlo trials.\n', mc_trials);

trial_results = cell(mc_trials, 1);
trial_seeds = seed0 + (0:(mc_trials - 1));

parfor trial_idx = 1:mc_trials
    trial_seed = trial_seeds(trial_idx);
    scene = func_init_constrained(trial_seed, M, sigma2, eps_s, reg_eps, initMaxTry, ...
        big_center, big_size, target_offset_z, target_size, vertex_box_center, ...
        vertex_cube_size, vertex_box_size, beta_max_deg);

    J0 = zeros(3, 3);
    sensor0 = repmat(struct( ...
        'd', NaN, ...
        'e', zeros(3, 1), ...
        'Pperp', zeros(3, 3), ...
        'mu', NaN, ...
        'sin_theta', NaN, ...
        'is_nondegenerate', false, ...
        'u', zeros(3, 1), ...
        'v', zeros(3, 1), ...
        'g', zeros(3, 1), ...
        'Ji', zeros(3, 3)), 1, M);
    for i = 1:M
        if isscalar(sigma2)
            sigma2_i = sigma2;
        else
            sigma2_i = sigma2(i);
        end
        [g_i, aux_i] = func_calc_jacobian(scene.P0(:, i), scene.U0(:, i), scene.x0, eps_s);
        Ji_i = (g_i * g_i.') / sigma2_i;
        aux_i.g = g_i;
        aux_i.Ji = Ji_i;
        sensor0(i) = aux_i;
        J0 = J0 + Ji_i;
    end
    J0 = (J0 + J0.') / 2;
    f0 = trace(inv(J0 + reg_eps * eye(size(J0))));

    out_gs_sqp = func_gs_sqp(scene, 'aopt', sigma2, eps_s, reg_eps, maxIter, stop_rel, beta_max_deg, gs_sqp);
    out_iso_sqp = func_iso_sqp(scene, 'aopt', sigma2, eps_s, reg_eps, maxIter, stop_rel, beta_max_deg, generic_mm);
    out_manopt = func_manopt(scene, trial_seed, 'aopt', sigma2, eps_s, reg_eps, manopt);
    out_tr_sqp = func_tr_sqp(scene, trial_seed, 'aopt', sigma2, eps_s, reg_eps, tr_sqp);
    out_pso = func_pso(scene, trial_seed, 'aopt', sigma2, eps_s, reg_eps, pso);
    out_de = func_de(scene, trial_seed, 'aopt', sigma2, eps_s, reg_eps, de);

    trial = struct();
    trial.trial_idx = trial_idx;
    trial.seed = trial_seed;
    trial.objective_init = f0;
    trial.min_sin_init = min([sensor0.sin_theta]);
    trial.gs_sqp = out_gs_sqp;
    trial.iso_sqp = out_iso_sqp;
    trial.manopt = out_manopt;
    trial.tr_sqp = out_tr_sqp;
    trial.pso = out_pso;
    trial.de = out_de;
    trial_results{trial_idx} = trial;
end

%% Tables
n_rows = mc_trials * n_methods;

scene_col = repmat("cube", n_rows, 1);
criterion_col = repmat("aopt", n_rows, 1);
trial_col = zeros(n_rows, 1);
seed_col = zeros(n_rows, 1);
method_id_col = strings(n_rows, 1);
method_name_col = strings(n_rows, 1);
f0_col = zeros(n_rows, 1);
ff_col = zeros(n_rows, 1);
reduction_col = zeros(n_rows, 1);
time_col = zeros(n_rows, 1);
iter_col = zeros(n_rows, 1);
min_eig_col = zeros(n_rows, 1);
min_sin0_col = zeros(n_rows, 1);
min_sinf_col = zeros(n_rows, 1);

row = 0;
for trial_idx = 1:mc_trials
    trial = trial_results{trial_idx};
    outs = {trial.gs_sqp, trial.iso_sqp, trial.manopt, trial.tr_sqp, trial.pso, trial.de};
    for method_idx = 1:n_methods
        row = row + 1;
        out = outs{method_idx};
        trial_col(row) = trial.trial_idx;
        seed_col(row) = trial.seed;
        method_id_col(row) = method_ids{method_idx};
        method_name_col(row) = method_names{method_idx};
        f0_col(row) = trial.objective_init;
        ff_col(row) = out.hist_f(end);
        reduction_col(row) = 100 * (trial.objective_init - out.hist_f(end)) / max(abs(trial.objective_init), 1e-12);
        time_col(row) = out.time_sec;
        iter_col(row) = out.T;
        min_eig_col(row) = out.hist_min_eigJ(end);
        min_sin0_col(row) = trial.min_sin_init;
        min_sinf_col(row) = out.hist_min_sin(end);
    end
end

trial_table = table(scene_col, criterion_col, trial_col, seed_col, method_id_col, method_name_col, ...
    f0_col, ff_col, reduction_col, time_col, iter_col, min_eig_col, min_sin0_col, min_sinf_col, ...
    'VariableNames', {'scene_name', 'criterion', 'trial_idx', 'seed', 'method_id', 'method_name', ...
    'objective_init', 'objective_final', 'reduction_pct', 'time_sec', 'iterations', ...
    'min_eigJ', 'min_sin_init', 'min_sin_final'});

summary_scene = repmat("cube", n_methods, 1);
summary_criterion = repmat("aopt", n_methods, 1);
summary_method_id = strings(n_methods, 1);
summary_method_name = strings(n_methods, 1);
summary_mc = mc_trials * ones(n_methods, 1);
f0_mean = zeros(n_methods, 1);
f0_std = zeros(n_methods, 1);
ff_mean = zeros(n_methods, 1);
ff_std = zeros(n_methods, 1);
reduction_mean = zeros(n_methods, 1);
reduction_std = zeros(n_methods, 1);
time_mean = zeros(n_methods, 1);
time_std = zeros(n_methods, 1);
iter_mean = zeros(n_methods, 1);
iter_std = zeros(n_methods, 1);
min_eig_mean = zeros(n_methods, 1);
min_sin0_mean = zeros(n_methods, 1);
min_sinf_mean = zeros(n_methods, 1);

for method_idx = 1:n_methods
    idx = trial_table.method_id == method_ids{method_idx};
    rows = trial_table(idx, :);
    summary_method_id(method_idx) = method_ids{method_idx};
    summary_method_name(method_idx) = method_names{method_idx};
    f0_mean(method_idx) = mean(rows.objective_init);
    f0_std(method_idx) = std(rows.objective_init);
    ff_mean(method_idx) = mean(rows.objective_final);
    ff_std(method_idx) = std(rows.objective_final);
    reduction_mean(method_idx) = mean(rows.reduction_pct);
    reduction_std(method_idx) = std(rows.reduction_pct);
    time_mean(method_idx) = mean(rows.time_sec);
    time_std(method_idx) = std(rows.time_sec);
    iter_mean(method_idx) = mean(rows.iterations);
    iter_std(method_idx) = std(rows.iterations);
    min_eig_mean(method_idx) = mean(rows.min_eigJ);
    min_sin0_mean(method_idx) = mean(rows.min_sin_init);
    min_sinf_mean(method_idx) = mean(rows.min_sin_final);
end

summary_table = table(summary_scene, summary_criterion, summary_method_id, summary_method_name, summary_mc, ...
    f0_mean, f0_std, ff_mean, ff_std, reduction_mean, reduction_std, ...
    time_mean, time_std, iter_mean, iter_std, min_eig_mean, min_sin0_mean, min_sinf_mean, ...
    'VariableNames', {'scene_name', 'criterion', 'method_id', 'method_name', 'mc_trials', ...
    'objective_init_mean', 'objective_init_std', 'objective_final_mean', 'objective_final_std', ...
    'reduction_pct_mean', 'reduction_pct_std', 'time_sec_mean', 'time_sec_std', ...
    'iterations_mean', 'iterations_std', 'min_eigJ_mean', 'min_sin_init_mean', 'min_sin_final_mean'});

disp(summary_table);

%% Figures
avg_gs_sqp = func_average_history(trial_results, 'gs_sqp');
avg_iso_sqp = func_average_history(trial_results, 'iso_sqp');
avg_manopt = func_average_history(trial_results, 'manopt');
avg_tr_sqp = func_average_history(trial_results, 'tr_sqp');
avg_pso = func_average_history(trial_results, 'pso');
avg_de = func_average_history(trial_results, 'de');

fig_history = figure('Color', 'w', 'Units', 'inches', 'Position', [1, 1, 5, 2.5]);
hold on; grid on; box on;
set(gca, 'FontSize', 8);
plot(0:(numel(avg_gs_sqp) - 1), avg_gs_sqp, 'k-', 'LineWidth', 1.6);
plot(0:(numel(avg_iso_sqp) - 1), avg_iso_sqp, '-.', 'Color', [0.85 0.10 0.10], 'LineWidth', 1.6);
plot(0:(numel(avg_manopt) - 1), avg_manopt, '-', 'Color', [0.47 0.67 0.19], 'LineWidth', 1.5);
plot(0:(numel(avg_tr_sqp) - 1), avg_tr_sqp, '--', 'Color', [0.30 0.75 0.93], 'LineWidth', 1.5);
plot(0:(numel(avg_pso) - 1), avg_pso, ':', 'Color', [0.49 0.18 0.56], 'LineWidth', 1.8);
plot(0:(numel(avg_de) - 1), avg_de, ':', 'Color', [0.93 0.69 0.13], 'LineWidth', 1.8);
xlabel('Outer sweep', 'FontSize', 8);
ylabel('tr(inv(J))', 'FontSize', 8);
title('constrained A-opt', 'FontSize', 8, 'FontWeight', 'normal', 'Interpreter', 'none');
legend(method_names, 'Location', 'best', 'FontSize', 7);

fig_runtime = figure('Color', 'w', 'Units', 'inches', 'Position', [1, 1, 5, 2.5]);
b = bar(summary_table.time_sec_mean, 0.6);
b.FaceColor = 'flat';
b.CData = [0.00 0.00 0.00; 0.85 0.10 0.10; 0.47 0.67 0.19; 0.30 0.75 0.93; 0.49 0.18 0.56; 0.93 0.69 0.13];
grid on; box on;
set(gca, 'XTick', 1:n_methods, 'XTickLabel', method_names);
set(gca, 'FontSize', 8);
ylabel('Mean runtime (s)', 'FontSize', 8);
title('constrained A-opt runtime', 'FontSize', 8, 'FontWeight', 'normal', 'Interpreter', 'none');

y_top = max(summary_table.time_sec_mean);
if y_top <= 0 || ~isfinite(y_top)
    y_top = 1;
end
ylim([0, 1.25 * y_top]);
for method_idx = 1:n_methods
    text(method_idx, summary_table.time_sec_mean(method_idx) + 0.03 * y_top, ...
        sprintf('%.3g', summary_table.time_sec_mean(method_idx)), ...
        'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', 'FontSize', 7);
end

results = struct();
results.case_name = 'constrained A-opt';
results.scene_name = 'cube';
results.criterion = 'aopt';
results.trial_results = trial_results;
results.trial_table = trial_table;
results.summary_table = summary_table;
results.mc_trials = mc_trials;

%% Save outputs
output_dir = fullfile(fileparts(mfilename('fullpath')), 'sim1');
if exist(output_dir, 'dir') ~= 7
    mkdir(output_dir);
end
writetable(trial_table, fullfile(output_dir, 'sim1_constrained_A_opt_trials.csv'));
writetable(summary_table, fullfile(output_dir, 'sim1_constrained_A_opt_summary.csv'));
savefig(fig_history, fullfile(output_dir, 'sim1_constrained_A_opt_history.fig'));
savefig(fig_runtime, fullfile(output_dir, 'sim1_constrained_A_opt_runtime.fig'));
func_save_figure(fig_history, fullfile(output_dir, 'sim1_constrained_A_opt_history.png'), 600);
func_save_figure(fig_runtime, fullfile(output_dir, 'sim1_constrained_A_opt_runtime.png'), 600);
results.history_fig = fullfile(output_dir, 'sim1_constrained_A_opt_history.fig');
results.runtime_fig = fullfile(output_dir, 'sim1_constrained_A_opt_runtime.fig');
results.history_png = fullfile(output_dir, 'sim1_constrained_A_opt_history.png');
results.runtime_png = fullfile(output_dir, 'sim1_constrained_A_opt_runtime.png');
save(fullfile(output_dir, 'sim1_constrained_A_opt_results.mat'), 'results', 'trial_table', 'summary_table', 'trial_results');

function avg_hist = func_average_history(trial_results, field_name)
	n_trials = numel(trial_results);
	hist_len = zeros(n_trials, 1);
	hist_cells = cell(n_trials, 1);

	for i = 1:n_trials
	    out = trial_results{i}.(field_name);
	    hist_cells{i} = out.hist_f(:).';
	    hist_len(i) = numel(hist_cells{i});
	end

	max_len = max(hist_len);
	hist_mat = nan(n_trials, max_len);
	for i = 1:n_trials
	    h = hist_cells{i};
	    hist_mat(i, 1:numel(h)) = h;
	    if numel(h) < max_len
	        hist_mat(i, (numel(h) + 1):max_len) = h(end);
	    end
	end

	avg_hist = mean(hist_mat, 1, 'omitnan');
end

function func_save_figure(fig, save_path, resolution)
	save_dir = fileparts(save_path);
	if exist(save_dir, 'dir') ~= 7
	    mkdir(save_dir);
	end
	try
	    exportgraphics(fig, save_path, 'Resolution', resolution);
	catch
	    saveas(fig, save_path);
	end
end





