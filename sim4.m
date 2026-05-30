clear;clc;close all;

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
vertex_box_size = [60; 60; 60];

beta_max_deg = 90;

delta_radii = 0:10:40;
num_random_directions = 1000;
delta_seed = seed0 + 1000;

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
gs_sqp.update_p = true;
gs_sqp.update_u = true;

output_dir = fullfile(fileparts(mfilename('fullpath')), 'sim4');
if exist(output_dir, 'dir') ~= 7
    mkdir(output_dir);
end

[delta_points, mismatch_radius, direction_index] = func_sample_mismatch_offsets( ...
    delta_radii, num_random_directions, delta_seed);

fprintf('Running sim4 nominal-target mismatch study.\n');

case_free_A = func_run_case('free_flight', 'Free-flight', 'aopt', seed0, ...
    delta_points, mismatch_radius, direction_index, M, sigma2, eps_s, reg_eps, ...
    maxIter, stop_rel, initMaxTry, big_center, free_space_size, constrained_space_size, ...
    target_offset_z, target_size, vertex_box_center, vertex_cube_size, vertex_box_size, ...
    beta_max_deg, gs_sqp);

case_free_D = func_run_case('free_flight', 'Free-flight', 'dopt', seed0, ...
    delta_points, mismatch_radius, direction_index, M, sigma2, eps_s, reg_eps, ...
    maxIter, stop_rel, initMaxTry, big_center, free_space_size, constrained_space_size, ...
    target_offset_z, target_size, vertex_box_center, vertex_cube_size, vertex_box_size, ...
    beta_max_deg, gs_sqp);

case_constrained_A = func_run_case('constrained', 'Constrained', 'aopt', seed0, ...
    delta_points, mismatch_radius, direction_index, M, sigma2, eps_s, reg_eps, ...
    maxIter, stop_rel, initMaxTry, big_center, free_space_size, constrained_space_size, ...
    target_offset_z, target_size, vertex_box_center, vertex_cube_size, vertex_box_size, ...
    beta_max_deg, gs_sqp);

case_constrained_D = func_run_case('constrained', 'Constrained', 'dopt', seed0, ...
    delta_points, mismatch_radius, direction_index, M, sigma2, eps_s, reg_eps, ...
    maxIter, stop_rel, initMaxTry, big_center, free_space_size, constrained_space_size, ...
    target_offset_z, target_size, vertex_box_center, vertex_cube_size, vertex_box_size, ...
    beta_max_deg, gs_sqp);

case_results = [case_free_A; case_free_D; case_constrained_A; case_constrained_D];
sample_table = vertcat( ...
    case_free_A.sample_table, ...
    case_free_D.sample_table, ...
    case_constrained_A.sample_table, ...
    case_constrained_D.sample_table);
summary_table = vertcat( ...
    case_free_A.summary_table, ...
    case_free_D.summary_table, ...
    case_constrained_A.summary_table, ...
    case_constrained_D.summary_table);

disp(summary_table);

fig = func_plot_degradation(case_results);

results = struct();
results.seed0 = seed0;
results.delta_radii = delta_radii;
results.num_random_directions = num_random_directions;
results.delta_points = delta_points;
results.mismatch_radius = mismatch_radius;
results.direction_index = direction_index;
results.case_results = case_results;
results.sample_table = sample_table;
results.summary_table = summary_table;

writetable(sample_table, fullfile(output_dir, 'sim4_samples.csv'));
writetable(summary_table, fullfile(output_dir, 'sim4_summary.csv'));
save(fullfile(output_dir, 'sim4_results.mat'), ...
    'results', 'case_results', 'sample_table', 'summary_table');
savefig(fig, fullfile(output_dir, 'sim4.fig'));
func_save_figure(fig, fullfile(output_dir, 'sim4.png'), 600);

fprintf('sim4 finished. Results saved in %s.\n', output_dir);

function case_result = func_run_case(scene_name, display_name, criterion, seed0, ...
    delta_points, mismatch_radius, direction_index, M, sigma2, eps_s, reg_eps, ...
    maxIter, stop_rel, initMaxTry, big_center, free_space_size, constrained_space_size, ...
    target_offset_z, target_size, vertex_box_center, vertex_cube_size, vertex_box_size, ...
    beta_max_deg, gs_sqp)

	fprintf('Optimizing %s %s.\n', display_name, func_criterion_title(criterion));
	scene = func_make_scene(scene_name, seed0, M, sigma2, eps_s, reg_eps, initMaxTry, ...
	    big_center, free_space_size, constrained_space_size, target_offset_z, target_size, ...
	    vertex_box_center, vertex_cube_size, vertex_box_size, beta_max_deg);

	out = func_gs_sqp(scene, criterion, sigma2, eps_s, reg_eps, ...
	    maxIter, stop_rel, beta_max_deg, gs_sqp);

	[J_nominal, sensor_nominal] = func_total_fim(out.P_final, out.U_final, out.x0, ...
	    sigma2, eps_s);
	objective_nominal = func_objective_value(J_nominal, criterion, reg_eps);

	sample_table = func_evaluate_mismatch(display_name, scene_name, criterion, out, ...
	    objective_nominal, delta_points, mismatch_radius, direction_index, ...
	    sigma2, eps_s, reg_eps);
	summary_table = func_summary_table(sample_table);

	case_result = struct();
	case_result.scene_name = scene_name;
	case_result.display_name = display_name;
	case_result.criterion = criterion;
	case_result.optimization = out;
	case_result.objective_nominal = objective_nominal;
	case_result.J_nominal = J_nominal;
	case_result.sensor_nominal = sensor_nominal;
	case_result.sample_table = sample_table;
	case_result.summary_table = summary_table;
end

function scene = func_make_scene(scene_name, seed0, M, sigma2, eps_s, reg_eps, initMaxTry, ...
    big_center, free_space_size, constrained_space_size, target_offset_z, target_size, ...
    vertex_box_center, vertex_cube_size, vertex_box_size, beta_max_deg)

	if strcmp(scene_name, 'free_flight')
	    scene = func_init_free(seed0, M, sigma2, eps_s, reg_eps, initMaxTry, ...
	        big_center, free_space_size, target_offset_z, target_size, beta_max_deg);
	else
	    scene = func_init_constrained(seed0, M, sigma2, eps_s, reg_eps, initMaxTry, ...
	        big_center, constrained_space_size, target_offset_z, target_size, ...
	        vertex_box_center, vertex_cube_size, vertex_box_size, beta_max_deg);
	end
end

function [delta_points, mismatch_radius, direction_index] = func_sample_mismatch_offsets( ...
    delta_radii, num_random_directions, delta_seed)

	delta_radii = sort(delta_radii(:).');
	n_total = 1 + num_random_directions * sum(delta_radii > 0);
	delta_points = zeros(3, n_total);
	mismatch_radius = zeros(n_total, 1);
	direction_index = ones(n_total, 1);

	rng(delta_seed);
	col = 1;
	for r_idx = 1:numel(delta_radii)
	    radius = delta_radii(r_idx);
	    if radius == 0
	        continue;
	    end

	    dirs = randn(3, num_random_directions);
	    dirs = dirs ./ vecnorm(dirs, 2, 1);
	    idx = col + (1:num_random_directions);
	    delta_points(:, idx) = radius * dirs;
	    mismatch_radius(idx) = radius;
	    direction_index(idx) = 1:num_random_directions;
	    col = col + num_random_directions;
	end

	delta_points = delta_points(:, 1:col);
	mismatch_radius = mismatch_radius(1:col);
	direction_index = direction_index(1:col);
end

function sample_table = func_evaluate_mismatch(display_name, scene_name, criterion, out, ...
    objective_nominal, delta_points, mismatch_radius, direction_index, sigma2, eps_s, reg_eps)

	n_samples = size(delta_points, 2);
	scene_col = strings(n_samples, 1);
	display_col = strings(n_samples, 1);
	criterion_col = strings(n_samples, 1);
	direction_col = zeros(n_samples, 1);
	radius_col = zeros(n_samples, 1);
	delta_x = zeros(n_samples, 1);
	delta_y = zeros(n_samples, 1);
	delta_z = zeros(n_samples, 1);
	x_true_x = zeros(n_samples, 1);
	x_true_y = zeros(n_samples, 1);
	x_true_z = zeros(n_samples, 1);
	objective_nominal_col = zeros(n_samples, 1);
	objective_true = zeros(n_samples, 1);
	normalized_degradation = zeros(n_samples, 1);
	min_eigJ = zeros(n_samples, 1);
	min_sin = zeros(n_samples, 1);

	for k = 1:n_samples
	    delta = delta_points(:, k);
	    x_true = out.x0 + delta;
	    [J_true, sensor_true] = func_total_fim(out.P_final, out.U_final, x_true, ...
	        sigma2, eps_s);
	    value_true = func_objective_value(J_true, criterion, reg_eps);

	    scene_col(k) = scene_name;
	    display_col(k) = display_name;
	    criterion_col(k) = criterion;
	    direction_col(k) = direction_index(k);
	    radius_col(k) = mismatch_radius(k);
	    delta_x(k) = delta(1);
	    delta_y(k) = delta(2);
	    delta_z(k) = delta(3);
	    x_true_x(k) = x_true(1);
	    x_true_y(k) = x_true(2);
	    x_true_z(k) = x_true(3);
	    objective_nominal_col(k) = objective_nominal;
	    objective_true(k) = value_true;
	    normalized_degradation(k) = value_true / objective_nominal;
	    min_eigJ(k) = min(eig((J_true + J_true.') / 2));
	    min_sin(k) = min([sensor_true.sin_theta]);
	end

	sample_table = table(scene_col, display_col, criterion_col, direction_col, radius_col, ...
	    delta_x, delta_y, delta_z, x_true_x, x_true_y, x_true_z, ...
	    objective_nominal_col, objective_true, normalized_degradation, min_eigJ, min_sin, ...
	    'VariableNames', {'scene_name', 'display_name', 'criterion', 'direction_index', ...
	    'mismatch_radius', 'delta_x', 'delta_y', 'delta_z', 'x_true_x', 'x_true_y', ...
	    'x_true_z', 'objective_nominal', 'objective_true', 'normalized_degradation', ...
	    'min_eigJ', 'min_sin'});
end

function summary_table = func_summary_table(sample_table)
	radii = unique(sample_table.mismatch_radius, 'stable');
	n_rows = numel(radii);

	scene_name = strings(n_rows, 1);
	display_name = strings(n_rows, 1);
	criterion = strings(n_rows, 1);
	mismatch_radius = zeros(n_rows, 1);
	sample_count = zeros(n_rows, 1);
	objective_nominal = zeros(n_rows, 1);
	normalized_degradation_mean = zeros(n_rows, 1);
	normalized_degradation_std = zeros(n_rows, 1);
	normalized_degradation_min = zeros(n_rows, 1);
	normalized_degradation_max = zeros(n_rows, 1);
	min_eigJ_mean = zeros(n_rows, 1);
	min_sin_mean = zeros(n_rows, 1);

	for i = 1:n_rows
	    idx = sample_table.mismatch_radius == radii(i);
	    rows = sample_table(idx, :);
	    scene_name(i) = rows.scene_name(1);
	    display_name(i) = rows.display_name(1);
	    criterion(i) = rows.criterion(1);
	    mismatch_radius(i) = radii(i);
	    sample_count(i) = height(rows);
	    objective_nominal(i) = rows.objective_nominal(1);
	    normalized_degradation_mean(i) = mean(rows.normalized_degradation);
	    normalized_degradation_std(i) = std(rows.normalized_degradation);
	    normalized_degradation_min(i) = min(rows.normalized_degradation);
	    normalized_degradation_max(i) = max(rows.normalized_degradation);
	    min_eigJ_mean(i) = mean(rows.min_eigJ);
	    min_sin_mean(i) = mean(rows.min_sin);
	end

	summary_table = table(scene_name, display_name, criterion, mismatch_radius, ...
	    sample_count, objective_nominal, normalized_degradation_mean, ...
	    normalized_degradation_std, normalized_degradation_min, ...
	    normalized_degradation_max, min_eigJ_mean, min_sin_mean, ...
	    'VariableNames', {'scene_name', 'display_name', 'criterion', 'mismatch_radius', ...
	    'sample_count', 'objective_nominal', 'normalized_degradation_mean', ...
	    'normalized_degradation_std', 'normalized_degradation_min', ...
	    'normalized_degradation_max', 'min_eigJ_mean', 'min_sin_mean'});
end

function fig = func_plot_degradation(case_results)
	font_size = 10;
	fig = figure('Color', 'w', 'Units', 'inches', 'Position', [1, 1, 7, 3.2]);

	pos_left = [0.10, 0.25, 0.38, 0.65];
	pos_right = [0.58, 0.25, 0.38, 0.65];

	ax1 = axes('Position', pos_left);
	func_plot_one_criterion(ax1, case_results, 'aopt', font_size);
	annotation('textbox', [pos_left(1), 0.08, pos_left(3), 0.06], ...
	    'String', '(a) A-opt', ...
	    'EdgeColor', 'none', ...
	    'HorizontalAlignment', 'center', ...
	    'FontSize', font_size);

	ax2 = axes('Position', pos_right);
	func_plot_one_criterion(ax2, case_results, 'dopt', font_size);
	annotation('textbox', [pos_right(1), 0.08, pos_right(3), 0.06], ...
	    'String', '(b) D-opt', ...
	    'EdgeColor', 'none', ...
	    'HorizontalAlignment', 'center', ...
	    'FontSize', font_size);

	legend(ax1, {'Free-flight', 'Constrained'}, 'Location', 'best', ...
	    'FontSize', font_size - 1, 'Box', 'on');
end

function func_plot_one_criterion(ax, case_results, criterion, font_size)
	hold(ax, 'on');
	grid(ax, 'on');
	box(ax, 'on');
	set(ax, 'FontSize', font_size);

	styles = struct();
	styles(1).scene_name = 'free_flight';
	styles(1).color = [0.00, 0.45, 0.74];
	styles(1).line_style = '-';
	styles(1).marker = 'o';
	styles(2).scene_name = 'constrained';
	styles(2).color = [0.85, 0.33, 0.10];
	styles(2).line_style = '--';
	styles(2).marker = 's';

	y_all = [];
	for i = 1:numel(styles)
	    idx_case = strcmp({case_results.scene_name}, styles(i).scene_name) & ...
	        strcmp({case_results.criterion}, criterion);
	    rows = case_results(idx_case).summary_table;
	    [x, order] = sort(rows.mismatch_radius);
	    y = rows.normalized_degradation_mean(order);
	    y_all = [y_all; y]; %#ok<AGROW>
	    plot(ax, x, y, ...
	        'Color', styles(i).color, ...
	        'LineStyle', styles(i).line_style, ...
	        'Marker', styles(i).marker, ...
	        'MarkerFaceColor', styles(i).color, ...
	        'MarkerSize', 5.5, ...
	        'LineWidth', 1.6);
	end

	xlabel(ax, 'Mismatch distance ||\Delta||_2 (m)', 'FontSize', font_size);
	ylabel(ax, 'Mean normalized degradation', 'FontSize', font_size);
	if strcmp(criterion, 'aopt')
	    title(ax, 'A-opt', 'FontWeight', 'normal', 'FontSize', font_size);
	else
	    title(ax, 'D-opt', 'FontWeight', 'normal', 'FontSize', font_size);
	end

	y_min = min(y_all);
	y_max = max(y_all);
	if abs(y_max - y_min) < 1e-12
	    ylim(ax, [y_min - 0.05, y_max + 0.05]);
	else
	    pad = 0.08 * (y_max - y_min);
	    ylim(ax, [y_min - pad, y_max + pad]);
	end
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

function txt = func_criterion_title(criterion)
	if strcmpi(criterion, 'aopt')
	    txt = 'A-opt';
	else
	    txt = 'D-opt';
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
