% sim1_plot.m

clear; clc; close all;

%% Configuration
results_dir = fullfile(fileparts(mfilename('fullpath')), 'sim1');

mat_files = {...
    'sim1_free_A_opt_results.mat', ...
    'sim1_free_D_opt_results.mat', ...
    'sim1_constrained_A_opt_results.mat', ...
    'sim1_constrained_D_opt_results.mat'};

subplot_labels = {'(a) free-flight A-opt', '(b) free-flight D-opt', ...
                  '(c) constrained A-opt', '(d) constrained D-opt'};

method_ids = {'gs_sqp', 'iso_sqp', 'manopt', 'tr_sqp', 'pso', 'de'};
method_names = {'GS-SQP', 'Iso-SQP', 'Manopt', 'TR-SQP', 'PSO', 'DE'};
n_methods = numel(method_ids);

% font size
font_size = 10;

% Line styles matching the individual sim1 scripts
line_styles = {'-', '-.', '-', '--', ':', ':'};
line_colors = {[0.00 0.00 0.00], [0.85 0.10 0.10], [0.47 0.67 0.19], ...
               [0.30 0.75 0.93], [0.49 0.18 0.56], [0.93 0.69 0.13]};
line_widths = [1.6, 1.6, 1.5, 1.5, 1.8, 1.8];

%% Create figure
fig = figure('Color', 'w', 'Units', 'inches', 'Position', [1, 1, 7, 6.5]);

% Custom subplot positions [left bottom width height]
% Use axes() instead of subplot() for precise control.
% MATLAB origin is bottom-left of figure; larger bottom = higher up.
margin_left  = 0.10;
margin_right = 0.04;
margin_top   = 0.04;
margin_bot   = 0.14;
gap_h = 0.09;
gap_v = 0.12;
col_w = (1 - margin_left - margin_right - gap_h) / 2;
row_h = (1 - margin_top - margin_bot - gap_v) / 2;

% Row 1 = bottom row (small y), Row 2 = top row (larger y)
pos_top_left  = [margin_left,             margin_bot + row_h + gap_v, col_w, row_h];
pos_top_right = [margin_left + col_w + gap_h, margin_bot + row_h + gap_v, col_w, row_h];
pos_bot_left  = [margin_left,             margin_bot,                  col_w, row_h];
pos_bot_right = [margin_left + col_w + gap_h, margin_bot,                  col_w, row_h];

% Position mapping: idx → [top-left, top-right, bottom-left, bottom-right]
plot_positions = {pos_top_left, pos_top_right, pos_bot_left, pos_bot_right};
ax_handles = gobjects(4, 1);

for idx = 1:4
    % Load MAT file
    S = load(fullfile(results_dir, mat_files{idx}));
    trial_results = S.trial_results;

    ax_handles(idx) = axes('Position', plot_positions{idx});
    hold on; grid on; box on;
    set(gca, 'FontSize', font_size);

    % Plot average convergence for each method
    for mi = 1:n_methods
        avg_hist = func_average_history(trial_results, method_ids{mi});
        plot(0:(numel(avg_hist) - 1), avg_hist, line_styles{mi}, ...
            'Color', line_colors{mi}, 'LineWidth', line_widths(mi));
    end

    % Y-label: A-opt on left column, D-opt on right column
    if mod(idx, 2) == 1
        ylabel('tr(inv(J))', 'FontSize', font_size);
    else
        ylabel('-log(det(J))', 'FontSize', font_size);
    end

    % X-label for all subplots
    xlabel('Iteration number', 'FontSize', font_size);

    % Subplot label as annotation below the axes (figure coordinates)
    pos_ax = plot_positions{idx};
    if idx <= 2
        label_y = pos_ax(2) - 0.09;  % top row: close to axis
    else
        label_y = pos_ax(2) - 0.09;  % bottom row
    end
    annotation('textbox', [pos_ax(1), label_y, pos_ax(3), 0.04], ...
        'String', subplot_labels{idx}, ...
        'EdgeColor', 'none', ...
        'HorizontalAlignment', 'center', ...
        'VerticalAlignment', 'top', ...
        'FontSize', font_size);
end

%% Legend in the top-left subplot (use saved handle, don't create new axes)
legend(ax_handles(1), method_names, 'Location', 'northeast', ...
    'FontSize', font_size - 2, 'Box', 'on');

%% Save outputs
output_dir = results_dir;
save_path_png = fullfile(output_dir, 'sim1_2x2_convergence.png');
save_path_pdf = fullfile(output_dir, 'sim1_2x2_convergence.pdf');
save_path_fig = fullfile(output_dir, 'sim1_2x2_convergence.fig');

func_save_figure(fig, save_path_png, 600);
func_save_figure(fig, save_path_pdf, 600);
savefig(fig, save_path_fig);

fprintf('Saved: %s\n', save_path_png);
fprintf('Saved: %s\n', save_path_pdf);

%% ========================================================================
%% Figure 2: Runtime bar charts (2×2)
%% ========================================================================
fig2 = figure('Color', 'w', 'Units', 'inches', 'Position', [1, 1, 7, 7]);

% Reuse same layout margins as convergence figure
ax2_handles = gobjects(4, 1);

for idx = 1:4
    S = load(fullfile(results_dir, mat_files{idx}));
    summary_table = S.summary_table;

    ax2_handles(idx) = axes('Position', plot_positions{idx});
    hold on; grid on; box on;
    set(gca, 'FontSize', font_size);

    % Extract runtime per method in the correct order
    time_vals = zeros(n_methods, 1);
    for mi = 1:n_methods
        row = strcmp(summary_table.method_id, method_ids{mi});
        time_vals(mi) = summary_table.time_sec_mean(row);
    end

    % Bar chart with method colors
    b = bar(time_vals, 0.6);
    b.FaceColor = 'flat';
    b.CData = vertcat(line_colors{:});

    % X-ticks: method names (rotated to fit)
    set(gca, 'XTick', 1:n_methods, 'XTickLabel', method_names);
    xtickangle(35);

    % Y-label
    ylabel('Mean runtime (s)', 'FontSize', font_size);

    % Subplot label (clears rotated x-tick labels)
    pos_ax = plot_positions{idx};
    if idx <= 2
        label_y = pos_ax(2) - 0.10;  % top row: close to axis
    else
        label_y = pos_ax(2) - 0.10;  % bottom row: room for rotated ticks
    end
    annotation('textbox', [pos_ax(1), label_y, pos_ax(3), 0.04], ...
        'String', subplot_labels{idx}, ...
        'EdgeColor', 'none', ...
        'HorizontalAlignment', 'center', ...
        'VerticalAlignment', 'top', ...
        'FontSize', font_size);

    % Value labels on top of bars
    y_top = max(time_vals);
    if y_top <= 0 || ~isfinite(y_top), y_top = 1; end
    ylim([0, 1.40 * y_top]);
    for mi = 1:n_methods
        text(mi, time_vals(mi) + 0.04 * y_top, ...
            sprintf('%.2f', time_vals(mi)), ...
            'HorizontalAlignment', 'center', ...
            'VerticalAlignment', 'bottom', ...
            'FontSize', font_size - 2);
    end
end

%% Save runtime figure
save_path_rt_png = fullfile(output_dir, 'sim1_2x2_runtime.png');
save_path_rt_pdf = fullfile(output_dir, 'sim1_2x2_runtime.pdf');
save_path_rt_fig = fullfile(output_dir, 'sim1_2x2_runtime.fig');

func_save_figure(fig2, save_path_rt_png, 600);
func_save_figure(fig2, save_path_rt_pdf, 600);
savefig(fig2, save_path_rt_fig);

fprintf('Saved: %s\n', save_path_rt_png);
fprintf('Saved: %s\n', save_path_rt_pdf);

%% ==================== Local functions ====================

function avg_hist = func_average_history(trial_results, field_name)
	% Compute the average convergence history across all Monte Carlo trials.
	% trial_results : cell array of structs, each having field_name struct
	% field_name    : string, e.g. 'gs_sqp', 'iso_sqp', ...
	%
	% avg_hist      : 1 x max_len row vector of averaged objective values

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
