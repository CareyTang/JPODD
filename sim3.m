clear;clc;close all;

mc_trials = 30;
seed0 = 55;

M = 8;
sigma2 = deg2rad(5)^2;
eps_s = 1e-2;
reg_eps = 1e-8;

maxIter = 100;
stop_rel = 1e-6;
initMaxTry = 80;

big_center = [0; 0; 0];
free_space_size = [500; 500; 100];
constrained_space_size = [500; 500; 300];
target_offset_z = 200;
target_size = [20; 20; 20];

vertex_box_center = big_center;
vertex_cube_size = [500; 500; 300];
vertex_box_size = [80; 80; 80];

beta_max_deg = 90;

% GS-SQP settings.
gs_sqp = struct();
gs_sqp.etaR0 = 1.0;
gs_sqp.etaT0 = 1.0;
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

trial_seeds = seed0 + (0:(mc_trials - 1));
output_dir = fullfile(fileparts(mfilename('fullpath')), 'sim3');
if exist(output_dir, 'dir') ~= 7
    mkdir(output_dir);
end

fprintf('Running sim3 GS-SQP importance study with %d Monte Carlo trials.\n', mc_trials);

case_free_A = func_run_case('free_flight', 'Free-flight A-opt', 'aopt', trial_seeds, ...
    M, sigma2, eps_s, reg_eps, maxIter, stop_rel, initMaxTry, ...
    big_center, free_space_size, constrained_space_size, target_offset_z, target_size, ...
    vertex_box_center, vertex_cube_size, vertex_box_size, beta_max_deg, gs_sqp);

case_free_D = func_run_case('free_flight', 'Free-flight D-opt', 'dopt', trial_seeds, ...
    M, sigma2, eps_s, reg_eps, maxIter, stop_rel, initMaxTry, ...
    big_center, free_space_size, constrained_space_size, target_offset_z, target_size, ...
    vertex_box_center, vertex_cube_size, vertex_box_size, beta_max_deg, gs_sqp);

case_constrained_A = func_run_case('constrained', 'Constrained A-opt', 'aopt', trial_seeds, ...
    M, sigma2, eps_s, reg_eps, maxIter, stop_rel, initMaxTry, ...
    big_center, free_space_size, constrained_space_size, target_offset_z, target_size, ...
    vertex_box_center, vertex_cube_size, vertex_box_size, beta_max_deg, gs_sqp);

case_constrained_D = func_run_case('constrained', 'Constrained D-opt', 'dopt', trial_seeds, ...
    M, sigma2, eps_s, reg_eps, maxIter, stop_rel, initMaxTry, ...
    big_center, free_space_size, constrained_space_size, target_offset_z, target_size, ...
    vertex_box_center, vertex_cube_size, vertex_box_size, beta_max_deg, gs_sqp);

case_results = [case_free_A; case_free_D; case_constrained_A; case_constrained_D];
trial_table = vertcat(case_results.trial_table);
summary_table = vertcat(case_results.summary_table);

disp(summary_table);

fig_history = func_plot_history(case_results);

results = struct();
results.mc_trials = mc_trials;
results.seed0 = seed0;
results.case_results = case_results;
results.trial_table = trial_table;
results.summary_table = summary_table;

writetable(trial_table, fullfile(output_dir, 'sim3_trials.csv'));
writetable(summary_table, fullfile(output_dir, 'sim3_summary.csv'));
save(fullfile(output_dir, 'sim3_results.mat'), ...
    'results', 'case_results', 'trial_table', 'summary_table');
savefig(fig_history, fullfile(output_dir, 'sim3.fig'));
func_save_figure(fig_history, fullfile(output_dir, 'sim3.png'), 600);

fprintf('sim3 finished. Results saved in %s.\n', output_dir);

function case_result = func_run_case(scene_name, case_label, criterion, trial_seeds, ...
    M, sigma2, eps_s, reg_eps, maxIter, stop_rel, initMaxTry, ...
    big_center, free_space_size, constrained_space_size, target_offset_z, target_size, ...
    vertex_box_center, vertex_cube_size, vertex_box_size, beta_max_deg, gs_sqp)
	n_trials = numel(trial_seeds);
	trial_results = cell(n_trials, 1);
	fprintf('Running %s with %d trials.\n', case_label, n_trials);

	for trial_idx = 1:n_trials
	    trial_seed = trial_seeds(trial_idx);
	    scene = func_make_scene(scene_name, trial_seed, M, sigma2, eps_s, reg_eps, initMaxTry, ...
	        big_center, free_space_size, constrained_space_size, target_offset_z, target_size, ...
	        vertex_box_center, vertex_cube_size, vertex_box_size, beta_max_deg);

	    [J0, sensor0] = func_total_fim(scene.P0, scene.U0, scene.x0, sigma2, eps_s);
	    objective_init = func_objective_value(J0, criterion, reg_eps);
	    min_sin_init = min([sensor0.sin_theta]);

	    gs_sqp_all = gs_sqp;
	    gs_sqp_all.update_p = true;
	    gs_sqp_all.update_u = true;
	    out_all = func_gs_sqp(scene, criterion, sigma2, eps_s, reg_eps, ...
	        maxIter, stop_rel, beta_max_deg, gs_sqp_all);
	    out_all.method_id = 'gs_sqp_all';
	    out_all.method_name = 'all';

	    gs_sqp_p_only = gs_sqp;
	    gs_sqp_p_only.update_p = true;
	    gs_sqp_p_only.update_u = false;
	    out_p_only = func_gs_sqp(scene, criterion, sigma2, eps_s, reg_eps, ...
	        maxIter, stop_rel, beta_max_deg, gs_sqp_p_only);
	    out_p_only.method_id = 'gs_sqp_p_only';
	    out_p_only.method_name = 'p-only';

	    gs_sqp_u_only = gs_sqp;
	    gs_sqp_u_only.update_p = false;
	    gs_sqp_u_only.update_u = true;
	    out_u_only = func_gs_sqp(scene, criterion, sigma2, eps_s, reg_eps, ...
	        maxIter, stop_rel, beta_max_deg, gs_sqp_u_only);
	    out_u_only.method_id = 'gs_sqp_u_only';
	    out_u_only.method_name = 'u-only';

	    trial = struct();
	    trial.trial_idx = trial_idx;
	    trial.seed = trial_seed;
	    trial.scene_name = scene_name;
	    trial.case_label = case_label;
	    trial.criterion = criterion;
	    trial.objective_init = objective_init;
	    trial.min_sin_init = min_sin_init;
	    trial.gs_sqp_all = out_all;
	    trial.gs_sqp_p_only = out_p_only;
	    trial.gs_sqp_u_only = out_u_only;
	    trial_results{trial_idx} = trial;

	    fprintf('mc:[%d]/%d %s.\n', trial_idx, n_trials, case_label);
	end

	trial_table = func_trial_table(trial_results);
	summary_table = func_summary_table(trial_table);

	case_result = struct();
	case_result.scene_name = scene_name;
	case_result.case_label = case_label;
	case_result.criterion = criterion;
	case_result.trial_results = {trial_results};
	case_result.trial_table = trial_table;
	case_result.summary_table = summary_table;
	case_result.avg_all = func_average_history(trial_results, 'gs_sqp_all');
	case_result.avg_p_only = func_average_history(trial_results, 'gs_sqp_p_only');
	case_result.avg_u_only = func_average_history(trial_results, 'gs_sqp_u_only');
end

function scene = func_make_scene(scene_name, seed, M, sigma2, eps_s, reg_eps, initMaxTry, ...
    big_center, free_space_size, constrained_space_size, target_offset_z, target_size, ...
    vertex_box_center, vertex_cube_size, vertex_box_size, beta_max_deg)
	if strcmp(scene_name, 'free_flight')
	    scene = func_init_free(seed, M, sigma2, eps_s, reg_eps, initMaxTry, ...
	        big_center, free_space_size, target_offset_z, target_size, beta_max_deg);
	else
	    scene = func_init_constrained(seed, M, sigma2, eps_s, reg_eps, initMaxTry, ...
	        big_center, constrained_space_size, target_offset_z, target_size, ...
	        vertex_box_center, vertex_cube_size, vertex_box_size, beta_max_deg);
	end
end

function trial_table = func_trial_table(trial_results)
	method_ids = {'gs_sqp_all', 'gs_sqp_p_only', 'gs_sqp_u_only'};
	method_names = {'all', 'p-only', 'u-only'};
	n_trials = numel(trial_results);
	n_methods = numel(method_ids);

	scene_name = strings(n_trials * n_methods, 1);
	case_label = strings(n_trials * n_methods, 1);
	criterion = strings(n_trials * n_methods, 1);
	method_id = strings(n_trials * n_methods, 1);
	method_name = strings(n_trials * n_methods, 1);
	trial_idx_col = zeros(n_trials * n_methods, 1);
	seed_col = zeros(n_trials * n_methods, 1);
	objective_init = zeros(n_trials * n_methods, 1);
	objective_final = zeros(n_trials * n_methods, 1);
	reduction_pct = zeros(n_trials * n_methods, 1);
	time_sec = zeros(n_trials * n_methods, 1);
	iterations = zeros(n_trials * n_methods, 1);
	min_eigJ = zeros(n_trials * n_methods, 1);
	min_sin_init = zeros(n_trials * n_methods, 1);
	min_sin_final = zeros(n_trials * n_methods, 1);

	row_idx = 0;
	for trial_idx = 1:n_trials
	    trial = trial_results{trial_idx};
	    for method_idx = 1:n_methods
	        row_idx = row_idx + 1;
	        out = trial.(method_ids{method_idx});
	        scene_name(row_idx) = trial.scene_name;
	        case_label(row_idx) = trial.case_label;
	        criterion(row_idx) = trial.criterion;
	        method_id(row_idx) = method_ids{method_idx};
	        method_name(row_idx) = method_names{method_idx};
	        trial_idx_col(row_idx) = trial.trial_idx;
	        seed_col(row_idx) = trial.seed;
	        objective_init(row_idx) = trial.objective_init;
	        objective_final(row_idx) = out.hist_f(end);
	        reduction_pct(row_idx) = 100 * ...
	            (trial.objective_init - out.hist_f(end)) / abs(trial.objective_init);
	        time_sec(row_idx) = out.time_sec;
	        iterations(row_idx) = out.T;
	        min_eigJ(row_idx) = out.hist_min_eigJ(end);
	        min_sin_init(row_idx) = trial.min_sin_init;
	        min_sin_final(row_idx) = out.hist_min_sin(end);
	    end
	end

	trial_table = table(scene_name, case_label, criterion, method_id, method_name, ...
	    trial_idx_col, seed_col, objective_init, objective_final, reduction_pct, ...
	    time_sec, iterations, min_eigJ, min_sin_init, min_sin_final, ...
	    'VariableNames', {'scene_name', 'case_label', 'criterion', 'method_id', 'method_name', ...
	    'trial_idx', 'seed', 'objective_init', 'objective_final', 'reduction_pct', ...
	    'time_sec', 'iterations', 'min_eigJ', 'min_sin_init', 'min_sin_final'});
end

function summary_table = func_summary_table(trial_table)
	method_ids = {'gs_sqp_all', 'gs_sqp_p_only', 'gs_sqp_u_only'};
	method_names = {'all', 'p-only', 'u-only'};
	n_methods = numel(method_ids);

	scene_name = strings(n_methods, 1);
	case_label = strings(n_methods, 1);
	criterion = strings(n_methods, 1);
	method_id = strings(n_methods, 1);
	method_name = strings(n_methods, 1);
	mc_trials = zeros(n_methods, 1);
	objective_init_mean = zeros(n_methods, 1);
	objective_init_std = zeros(n_methods, 1);
	objective_final_mean = zeros(n_methods, 1);
	objective_final_std = zeros(n_methods, 1);
	reduction_pct_mean = zeros(n_methods, 1);
	reduction_pct_std = zeros(n_methods, 1);
	time_sec_mean = zeros(n_methods, 1);
	time_sec_std = zeros(n_methods, 1);
	iterations_mean = zeros(n_methods, 1);
	iterations_std = zeros(n_methods, 1);
	min_eigJ_mean = zeros(n_methods, 1);
	min_sin_init_mean = zeros(n_methods, 1);
	min_sin_final_mean = zeros(n_methods, 1);

	for method_idx = 1:n_methods
	    rows = trial_table(trial_table.method_id == method_ids{method_idx}, :);
	    scene_name(method_idx) = rows.scene_name(1);
	    case_label(method_idx) = rows.case_label(1);
	    criterion(method_idx) = rows.criterion(1);
	    method_id(method_idx) = method_ids{method_idx};
	    method_name(method_idx) = method_names{method_idx};
	    mc_trials(method_idx) = height(rows);
	    objective_init_mean(method_idx) = mean(rows.objective_init);
	    objective_init_std(method_idx) = std(rows.objective_init);
	    objective_final_mean(method_idx) = mean(rows.objective_final);
	    objective_final_std(method_idx) = std(rows.objective_final);
	    reduction_pct_mean(method_idx) = mean(rows.reduction_pct);
	    reduction_pct_std(method_idx) = std(rows.reduction_pct);
	    time_sec_mean(method_idx) = mean(rows.time_sec);
	    time_sec_std(method_idx) = std(rows.time_sec);
	    iterations_mean(method_idx) = mean(rows.iterations);
	    iterations_std(method_idx) = std(rows.iterations);
	    min_eigJ_mean(method_idx) = mean(rows.min_eigJ);
	    min_sin_init_mean(method_idx) = mean(rows.min_sin_init);
	    min_sin_final_mean(method_idx) = mean(rows.min_sin_final);
	end

	summary_table = table(scene_name, case_label, criterion, method_id, method_name, mc_trials, ...
	    objective_init_mean, objective_init_std, objective_final_mean, objective_final_std, ...
	    reduction_pct_mean, reduction_pct_std, time_sec_mean, time_sec_std, ...
	    iterations_mean, iterations_std, min_eigJ_mean, min_sin_init_mean, min_sin_final_mean, ...
	    'VariableNames', {'scene_name', 'case_label', 'criterion', 'method_id', 'method_name', ...
	    'mc_trials', 'objective_init_mean', 'objective_init_std', ...
	    'objective_final_mean', 'objective_final_std', 'reduction_pct_mean', ...
	    'reduction_pct_std', 'time_sec_mean', 'time_sec_std', ...
	    'iterations_mean', 'iterations_std', 'min_eigJ_mean', ...
	    'min_sin_init_mean', 'min_sin_final_mean'});
end

function fig = func_plot_history(case_results)
	method_names = {'all', 'p-only', 'u-only'};
	subplot_labels = {'(a) free-flight A-opt', '(b) free-flight D-opt', ...
	    '(c) constrained A-opt', '(d) constrained D-opt'};
	line_styles = {'-', '--', ':'};
	line_colors = {[0.00 0.00 0.00], [0.85 0.10 0.10], [0.00 0.45 0.74]};
	line_widths = [1.7, 1.7, 1.9];
	font_size = 10;

	fig = figure('Color', 'w', 'Units', 'inches', 'Position', [1, 1, 7, 6.5]);

	margin_left  = 0.10;
	margin_right = 0.04;
	margin_top   = 0.04;
	margin_bot   = 0.14;
	gap_h = 0.09;
	gap_v = 0.12;
	col_w = (1 - margin_left - margin_right - gap_h) / 2;
	row_h = (1 - margin_top - margin_bot - gap_v) / 2;

	pos_top_left  = [margin_left, margin_bot + row_h + gap_v, col_w, row_h];
	pos_top_right = [margin_left + col_w + gap_h, margin_bot + row_h + gap_v, col_w, row_h];
	pos_bot_left  = [margin_left, margin_bot, col_w, row_h];
	pos_bot_right = [margin_left + col_w + gap_h, margin_bot, col_w, row_h];
	plot_positions = {pos_top_left, pos_top_right, pos_bot_left, pos_bot_right};
	ax_handles = gobjects(4, 1);

	for case_idx = 1:4
	    ax_handles(case_idx) = axes('Position', plot_positions{case_idx});
	    hold on; grid on; box on;
	    set(gca, 'FontSize', font_size);

	    histories = {case_results(case_idx).avg_all, ...
	        case_results(case_idx).avg_p_only, ...
	        case_results(case_idx).avg_u_only};
	    for method_idx = 1:3
	        avg_hist = histories{method_idx};
	        plot(0:(numel(avg_hist) - 1), avg_hist, line_styles{method_idx}, ...
	            'Color', line_colors{method_idx}, 'LineWidth', line_widths(method_idx));
	    end

	    if strcmp(case_results(case_idx).criterion, 'aopt')
	        ylabel('tr(inv(J))', 'FontSize', font_size);
	    else
	        ylabel('-log(det(J))', 'FontSize', font_size);
	    end
	    xlabel('Iteration number', 'FontSize', font_size);

	    pos_ax = plot_positions{case_idx};
	    annotation('textbox', [pos_ax(1), pos_ax(2) - 0.09, pos_ax(3), 0.04], ...
	        'String', subplot_labels{case_idx}, ...
	        'EdgeColor', 'none', ...
	        'HorizontalAlignment', 'center', ...
	        'VerticalAlignment', 'top', ...
	        'FontSize', font_size);
	end

	legend(ax_handles(1), method_names, 'Location', 'northeast', ...
	    'FontSize', font_size - 1, 'Box', 'on');
end

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

function [J, sensorData] = func_total_fim(P, U, x0, sigma2, eps_s)
	M = size(P, 2);
	J = zeros(3, 3);
	sensorData = repmat(func_sensor_template(), 1, M);
	for i = 1:M
	    if isscalar(sigma2)
	        sigma2_i = sigma2;
	    else
	        sigma2_i = sigma2(i);
	    end
	    [g_i, aux_i] = func_calc_jacobian(P(:, i), U(:, i), x0, eps_s);
	    aux_i.g = g_i;
	    aux_i.Ji = (g_i * g_i.') / sigma2_i;
	    sensorData(i) = aux_i;
	    J = J + aux_i.Ji;
	end
	J = (J + J.') / 2;
end

function f = func_objective_value(J, criterion, reg_eps)
	J = (J + J.') / 2;
	if strcmpi(criterion, 'aopt')
	    f = trace(inv(J + reg_eps * eye(3)));
	else
	    f = -log(det(J + reg_eps * eye(3)));
	end
end

function tmpl = func_sensor_template()
	tmpl = struct('d', NaN, 'e', zeros(3, 1), 'Pperp', zeros(3, 3), ...
	    'mu', NaN, 'sin_theta', NaN, 'is_nondegenerate', false, ...
	    'u', zeros(3, 1), 'v', zeros(3, 1), 'g', zeros(3, 1), 'Ji', zeros(3, 3));
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
