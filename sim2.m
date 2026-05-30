clear;clc;close all;

mc_trials = 1;
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
vertex_target_offset_z = 0;
target_size = [20; 20; 20];

vertex_box_center = big_center;
vertex_cube_size = [500; 500; 300];
vertex_box_size = [60; 60; 60];

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

output_dir = fullfile(fileparts(mfilename('fullpath')), 'sim2');
if exist(output_dir, 'dir') ~= 7
    mkdir(output_dir);
end

fprintf('Running sim2 with %d Monte Carlo trial.\n', mc_trials);

trial_seed = seed0;

scene_free_a = func_init_free(trial_seed, M, sigma2, eps_s, reg_eps, initMaxTry, ...
    big_center, free_space_size, target_offset_z, target_size, beta_max_deg);
out_free_a = func_gs_sqp(scene_free_a, 'aopt', sigma2, eps_s, reg_eps, ...
    maxIter, stop_rel, beta_max_deg, gs_sqp);

scene_free_d = func_init_free(trial_seed + 1, M, sigma2, eps_s, reg_eps, initMaxTry, ...
    big_center, free_space_size, target_offset_z, target_size, beta_max_deg);
out_free_d = func_gs_sqp(scene_free_d, 'dopt', sigma2, eps_s, reg_eps, ...
    maxIter, stop_rel, beta_max_deg, gs_sqp);

scene_constrained_a = func_init_constrained(trial_seed + 2, M, sigma2, eps_s, reg_eps, ...
    initMaxTry, big_center, constrained_space_size, vertex_target_offset_z, target_size, ...
    vertex_box_center, vertex_cube_size, vertex_box_size, beta_max_deg);
out_constrained_a = func_gs_sqp(scene_constrained_a, 'aopt', sigma2, eps_s, reg_eps, ...
    maxIter, stop_rel, beta_max_deg, gs_sqp);

scene_constrained_d = func_init_constrained(trial_seed + 3, M, sigma2, eps_s, reg_eps, ...
    initMaxTry, big_center, constrained_space_size, vertex_target_offset_z, target_size, ...
    vertex_box_center, vertex_cube_size, vertex_box_size, beta_max_deg);
out_constrained_d = func_gs_sqp(scene_constrained_d, 'dopt', sigma2, eps_s, reg_eps, ...
    maxIter, stop_rel, beta_max_deg, gs_sqp);

fig = figure('Color', 'w', 'Units', 'inches', 'Position', [1, 1, 12, 16], 'Visible', 'on');
tl = tiledlayout(fig, 4, 3, 'Padding', 'loose', 'TileSpacing', 'loose');

ax1 = nexttile(tl, 1);
func_plot_geometry(ax1, out_free_a, 'Free-flight | A-opt', 'Full view', ...
    '(a1)', true, false);

ax2 = nexttile(tl, 2);
func_plot_geometry(ax2, out_free_a, 'Free-flight | A-opt', 'XZ view', '(a2)', true, false);

ax3 = nexttile(tl, 3);
func_plot_geometry(ax3, out_free_a, 'Free-flight | A-opt', 'XY view', '(a3)', true, false);

ax4 = nexttile(tl, 4);
func_plot_geometry(ax4, out_free_d, 'Free-flight | D-opt', 'Full view', '(b1)', true, false);

ax5 = nexttile(tl, 5);
func_plot_geometry(ax5, out_free_d, 'Free-flight | D-opt', 'XZ view', '(b2)', true, false);

ax6 = nexttile(tl, 6);
func_plot_geometry(ax6, out_free_d, 'Free-flight | D-opt', 'XY view', '(b3)', true, false);

ax7 = nexttile(tl, 7);
func_plot_geometry(ax7, out_constrained_a, 'Constrained | A-opt', 'Full view', '(c1)', false, true);

ax8 = nexttile(tl, 8);
func_plot_geometry(ax8, out_constrained_a, 'Constrained | A-opt', 'XZ view', '(c2)', false, true);

ax9 = nexttile(tl, 9);
func_plot_geometry(ax9, out_constrained_a, 'Constrained | A-opt', 'XY view', '(c3)', false, true);

ax10 = nexttile(tl, 10);
func_plot_geometry(ax10, out_constrained_d, 'Constrained | D-opt', 'Full view', '(d1)', false, true);

ax11 = nexttile(tl, 11);
func_plot_geometry(ax11, out_constrained_d, 'Constrained | D-opt', 'XZ view', '(d2)', false, true);

ax12 = nexttile(tl, 12);
func_plot_geometry(ax12, out_constrained_d, 'Constrained | D-opt', 'XY view', '(d3)', false, true);

legend_ax = axes(fig, 'Position', [0.12, 0.012, 0.76, 0.025], 'Visible', 'off');
hold(legend_ax, 'on');
h_target_lgd = plot(legend_ax, nan, nan, 'kp', ...
    'MarkerSize', 10, 'MarkerFaceColor', 'w', 'LineWidth', 1.4);
h_start_lgd = plot(legend_ax, nan, nan, 'ko', ...
    'MarkerSize', 6, 'LineWidth', 1.2);
h_start_u_lgd = plot(legend_ax, nan, nan, 'k-', 'LineWidth', 1.2);
h_end_lgd = plot(legend_ax, nan, nan, 'ro', ...
    'MarkerSize', 6, 'LineWidth', 1.2);
h_end_u_lgd = plot(legend_ax, nan, nan, 'r-', 'LineWidth', 1.2);
legend(legend_ax, [h_target_lgd, h_start_lgd, h_start_u_lgd, h_end_lgd, h_end_u_lgd], ...
    {'Target', 'Start points', 'Start orientations', 'End points', 'End orientations'}, ...
    'Orientation', 'horizontal', 'Location', 'north', 'Box', 'on', 'FontSize', 10);

results = struct();
results.mc_trials = mc_trials;
results.seed0 = seed0;
results.free_A_opt = out_free_a;
results.free_D_opt = out_free_d;
results.constrained_A_opt = out_constrained_a;
results.constrained_D_opt = out_constrained_d;

save(fullfile(output_dir, 'sim2_results.mat'), 'results');
savefig(fig, fullfile(output_dir, 'sim2_geometry.fig'));
func_save_figure(fig, fullfile(output_dir, 'sim2_geometry.png'), 600);

fprintf('sim2 finished. Results saved in %s.\n', output_dir);

function legend_handles = func_plot_geometry(ax, out, case_title, view_type, panel_label, ...
    draw_outer_box, draw_sensor_boxes)
	hold(ax, 'on');
	grid(ax, 'on');
	box(ax, 'on');
	axis(ax, 'equal');

	if strcmpi(view_type, 'XZ view')
	    view(ax, [0, 0]);
	elseif strcmpi(view_type, 'XY view')
	    view(ax, [0, 90]);
	else
	    view(ax, [38, 24]);
	end

	if draw_outer_box
	    func_draw_box_wire(ax, out.lb, out.ub, [0.72, 0.72, 0.72], 0.9);
	end
	if draw_sensor_boxes
	    func_draw_sensor_boxes(ax, out.P_lb_each, out.P_ub_each);
	end

	plot3(ax, out.x0(1), out.x0(2), out.x0(3), 'kp', ...
	    'MarkerSize', 14, 'MarkerFaceColor', 'w', 'LineWidth', 1.6);

	for i = 1:size(out.P_final, 2)
	    traj = squeeze(out.P_hist(:, i, :));
	    plot3(ax, traj(1, :), traj(2, :), traj(3, :), '-', ...
	        'Color', [0.72, 0.72, 0.72], 'LineWidth', 0.9);
	end

	h_target = plot3(ax, out.x0(1), out.x0(2), out.x0(3), 'kp', ...
	    'MarkerSize', 14, 'MarkerFaceColor', 'w', 'LineWidth', 1.6);
	h_start = plot3(ax, out.P_init(1, :), out.P_init(2, :), out.P_init(3, :), ...
	    'ko', 'MarkerSize', 6, 'LineWidth', 1.2);
	h_end = plot3(ax, out.P_final(1, :), out.P_final(2, :), out.P_final(3, :), ...
	    'ro', 'MarkerSize', 6, 'LineWidth', 1.2);

	U_end = func_align_signs(out.U_final, out.U_init);
	arrow_scale = 0.16 * norm(out.ub - out.lb);
	h_start_u = quiver3(ax, out.P_init(1, :), out.P_init(2, :), out.P_init(3, :), ...
	    arrow_scale * out.U_init(1, :), arrow_scale * out.U_init(2, :), ...
	    arrow_scale * out.U_init(3, :), 0, 'Color', 'k', 'LineWidth', 1.1, ...
	    'MaxHeadSize', 1.2);
	h_end_u = quiver3(ax, out.P_final(1, :), out.P_final(2, :), out.P_final(3, :), ...
	    arrow_scale * U_end(1, :), arrow_scale * U_end(2, :), ...
	    arrow_scale * U_end(3, :), 0, 'Color', 'r', 'LineWidth', 1.1, ...
	    'MaxHeadSize', 1.2);

	axis_lb = min([out.P_lb_each, out.x0], [], 2);
	axis_ub = max([out.P_ub_each, out.x0], [], 2);
	axis_margin = 0.06 * max(axis_ub - axis_lb);
	xlim(ax, [axis_lb(1) - axis_margin, axis_ub(1) + axis_margin]);
	ylim(ax, [axis_lb(2) - axis_margin, axis_ub(2) + axis_margin]);
	zlim(ax, [axis_lb(3) - axis_margin, axis_ub(3) + axis_margin]);
	if strcmpi(view_type, 'XZ view') || strcmpi(view_type, 'XY view')
	    axis(ax, 'normal');
	    pbaspect(ax, [1, 1, 1]);
	end

	func_add_bottom_title(ax, sprintf('%s %s | %s', panel_label, case_title, view_type));
	legend_handles = [h_target, h_start, h_start_u, h_end, h_end_u];
end

function func_draw_sensor_boxes(ax, lb_each, ub_each)
	M = size(lb_each, 2);
	for i = 1:M
	    func_draw_cube(ax, lb_each(:, i), ub_each(:, i));
	end
end

function func_draw_cube(ax, lb, ub)
	verts = [ ...
	    lb(1), lb(2), lb(3);
	    ub(1), lb(2), lb(3);
	    ub(1), ub(2), lb(3);
	    lb(1), ub(2), lb(3);
	    lb(1), lb(2), ub(3);
	    ub(1), lb(2), ub(3);
	    ub(1), ub(2), ub(3);
	    lb(1), ub(2), ub(3)];
	faces = [ ...
	    1 2 3 4;
	    5 6 7 8;
	    1 2 6 5;
	    2 3 7 6;
	    3 4 8 7;
	    4 1 5 8];

	patch(ax, 'Vertices', verts, 'Faces', faces, ...
	    'FaceColor', [0.88, 0.88, 0.88], ...
	    'FaceAlpha', 0.16, ...
	    'EdgeColor', [0.62, 0.62, 0.62], ...
	    'LineWidth', 0.9);
end

function func_draw_box_wire(ax, lb, ub, color_rgb, line_width)
	verts = [ ...
	    lb(1), lb(2), lb(3);
	    ub(1), lb(2), lb(3);
	    ub(1), ub(2), lb(3);
	    lb(1), ub(2), lb(3);
	    lb(1), lb(2), ub(3);
	    ub(1), lb(2), ub(3);
	    ub(1), ub(2), ub(3);
	    lb(1), ub(2), ub(3)];
	edges = [ ...
	    1 2; 2 3; 3 4; 4 1;
	    5 6; 6 7; 7 8; 8 5;
	    1 5; 2 6; 3 7; 4 8];

	for k = 1:size(edges, 1)
	    pts = verts(edges(k, :), :);
	    plot3(ax, pts(:, 1), pts(:, 2), pts(:, 3), '-', ...
	        'Color', color_rgb, 'LineWidth', line_width);
	end
end

function U_aligned = func_align_signs(U, U_ref)
	U_aligned = U;
	for i = 1:size(U, 2)
	    if dot(U_aligned(:, i), U_ref(:, i)) < 0
	        U_aligned(:, i) = -U_aligned(:, i);
	    end
	end
end

function func_add_bottom_title(ax, label_text)
	text(ax, 0.5, -0.13, label_text, ...
	    'Units', 'normalized', ...
	    'HorizontalAlignment', 'center', ...
	    'VerticalAlignment', 'top', ...
	    'FontSize', 10, ...
	    'FontWeight', 'normal', ...
	    'Clipping', 'off');
end

function func_save_figure(fig, save_path, resolution)
	try
	    exportgraphics(fig, save_path, 'Resolution', resolution);
	catch
	    print(fig, save_path, '-dpng', sprintf('-r%d', resolution));
	end
end
