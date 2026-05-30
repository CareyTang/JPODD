clear;clc;close all;

big_center = [0; 0; 0];
free_space_size = [500; 500; 100];
cube_size = [500; 500; 300];
cube_local_size = [60; 60; 60];
target_position = big_center + [0; 0; 200];
figure_position = [120, 120, 1120, 500];
panel_view = [-35, 20];

output_dir = fullfile(fileparts(mfilename('fullpath')), 'sim0');
output_pdf_name = 'simulation_region.pdf';
output_png_name = 'simulation_region.png';
export_resolution = 800;

free_scene = func_build_free_space_scene(big_center, free_space_size, target_position);
cube_scene = func_build_cube_scene(big_center, cube_size, cube_local_size, target_position);

fig = figure( ...
    'Color', 'w', ...
    'Visible', 'on', ...
    'Position', figure_position);
tl = tiledlayout(fig, 1, 2, 'Padding', 'compact', 'TileSpacing', 'compact');

ax_free = nexttile(tl, 1);
func_plot_free_space_panel(ax_free, free_scene, panel_view);

ax_cube = nexttile(tl, 2);
func_plot_cube_panel(ax_cube, cube_scene, panel_view);

results = struct();
results.big_center = big_center;
results.free_space_size = free_space_size;
results.cube_size = cube_size;
results.cube_local_size = cube_local_size;
results.target_position = target_position;
results.fig = fig;
results.free_scene = free_scene;
results.cube_scene = cube_scene;

results.output_paths.pdf = fullfile(output_dir, output_pdf_name);
results.output_paths.png = fullfile(output_dir, output_png_name);
func_save_figure(fig, results.output_paths.pdf, export_resolution);
func_save_figure(fig, results.output_paths.png, export_resolution);

function scene = func_build_free_space_scene(big_center, free_space_size, target_position)
	    scene = struct();
	    scene.center = big_center(:);
	    scene.lb = big_center(:) - 0.5 * free_space_size(:);
	    scene.ub = big_center(:) + 0.5 * free_space_size(:);
	    scene.target = target_position(:);
end

function scene = func_build_cube_scene(big_center, cube_size, cube_local_size, target_position)
	    vertex_signs = [ ...
	        -1, -1, -1;
	        -1, -1,  1;
	        -1,  1, -1;
	        -1,  1,  1;
	        1, -1, -1;
	        1, -1,  1;
	        1,  1, -1;
	        1,  1,  1].';
	    centers = big_center(:) + 0.5 * cube_size(:) .* vertex_signs;
	    lb_each = centers - 0.5 * cube_local_size(:);
	    ub_each = centers + 0.5 * cube_local_size(:);

	    scene = struct();
	    scene.center = big_center(:);
	    scene.lb = big_center(:) - 0.5 * cube_size(:);
	    scene.ub = big_center(:) + 0.5 * cube_size(:);
	    scene.lb_each = lb_each;
	    scene.ub_each = ub_each;
	    scene.centers = centers;
	    scene.target = target_position(:);
end

function func_plot_free_space_panel(ax, scene, panel_view)
	    hold(ax, 'on');
	    axis(ax, 'equal');
	    view(ax, panel_view);
	    box(ax, 'on');
	    grid(ax, 'off');

	    func_draw_box_patch(ax, scene.lb, scene.ub, ...
	        [0.77, 0.90, 0.98], 0.18, [0.16, 0.43, 0.67], 1.4);
	    func_draw_target_marker(ax, scene.center, scene.target);

	    text(ax, scene.lb(1) + 15, scene.ub(2) - 35, scene.ub(3) + 12, ...
	        'Deployment region', ...
	        'Color', [0.16, 0.43, 0.67], ...
	        'FontWeight', 'bold');

	    title(ax, 'Free-flight region', 'FontWeight', 'bold');
	    func_format_axes(ax, scene.lb, scene.ub, scene.target);
end

function func_plot_cube_panel(ax, scene, panel_view)
	    hold(ax, 'on');
	    axis(ax, 'equal');
	    view(ax, panel_view);
	    box(ax, 'on');
	    grid(ax, 'off');

	    func_draw_box_wire(ax, scene.lb, scene.ub, [0.65, 0.65, 0.65], 1.0, '--');
	    for i = 1:size(scene.lb_each, 2)
	        func_draw_box_patch(ax, scene.lb_each(:, i), scene.ub_each(:, i), ...
	            [0.98, 0.84, 0.68], 0.32, [0.74, 0.41, 0.15], 1.0);
	    end
	    func_draw_target_marker(ax, scene.center, scene.target);

	    text(ax, scene.centers(1, 8) - 40, scene.centers(2, 8) - 10, scene.centers(3, 8) + 55, ...
	        '8 local cubes', ...
	        'Color', [0.62, 0.31, 0.08], ...
	        'FontWeight', 'bold');

	    title(ax, 'Constrained hovering regions', 'FontWeight', 'bold');
	    func_format_axes(ax, scene.lb, scene.ub, scene.target);
end

function func_format_axes(ax, lb, ub, target)
	    pad = [50; 50; 35];
	    mins = min([lb, target], [], 2) - pad;
	    maxs = max([ub, target], [], 2) + pad;

	    xlim(ax, [mins(1), maxs(1)]);
	    ylim(ax, [mins(2), maxs(2)]);
	    zlim(ax, [mins(3), maxs(3)]);

	    xlabel(ax, 'x (m)');
	    ylabel(ax, 'y (m)');
	    zlabel(ax, 'z (m)');
	    set(ax, ...
	        'FontName', 'Times New Roman', ...
	        'LineWidth', 0.8, ...
	        'XTickMode', 'auto', ...
	        'YTickMode', 'auto', ...
	        'ZTickMode', 'auto');
end

function func_draw_target_marker(ax, center, target)
	    plot3(ax, [center(1), target(1)], [center(2), target(2)], [center(3), target(3)], ...
	        '--', 'Color', [0.35, 0.35, 0.35], 'LineWidth', 1.0);
	    plot3(ax, target(1), target(2), target(3), 'p', ...
	        'MarkerSize', 12, ...
	        'MarkerFaceColor', [0.86, 0.29, 0.19], ...
	        'MarkerEdgeColor', [0.55, 0.10, 0.08], ...
	        'LineWidth', 1.1);
	    text(ax, target(1) + 12, target(2) + 12, target(3) + 6, ...
	        'Target x_0', ...
	        'Color', [0.55, 0.10, 0.08], ...
	        'FontWeight', 'bold');
end

function func_draw_box_patch(ax, lb, ub, face_color, face_alpha, edge_color, line_width)
	    [verts, faces, edges] = func_box_geometry(lb, ub);

	    patch(ax, ...
	        'Vertices', verts, ...
	        'Faces', faces, ...
	        'FaceColor', face_color, ...
	        'FaceAlpha', face_alpha, ...
	        'EdgeColor', 'none');

	    for e = 1:size(edges, 1)
	        pts = verts(edges(e, :), :);
	        plot3(ax, pts(:, 1), pts(:, 2), pts(:, 3), ...
	            '-', 'Color', edge_color, 'LineWidth', line_width);
	    end
end

function func_draw_box_wire(ax, lb, ub, edge_color, line_width, line_style)
	    [verts, ~, edges] = func_box_geometry(lb, ub);
	    for e = 1:size(edges, 1)
	        pts = verts(edges(e, :), :);
	        plot3(ax, pts(:, 1), pts(:, 2), pts(:, 3), ...
	            'LineStyle', line_style, 'Color', edge_color, 'LineWidth', line_width);
	    end
end

function [verts, faces, edges] = func_box_geometry(lb, ub)
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
	        1, 2, 3, 4;
	        5, 6, 7, 8;
	        1, 2, 6, 5;
	        2, 3, 7, 6;
	        3, 4, 8, 7;
	        4, 1, 5, 8];

	    edges = [ ...
	        1, 2; 2, 3; 3, 4; 4, 1;
	        5, 6; 6, 7; 7, 8; 8, 5;
	        1, 5; 2, 6; 3, 7; 4, 8];
end

function func_save_figure(fig, save_path, resolution)
	    [save_dir, ~, ~] = fileparts(save_path);
	    if exist(save_dir, 'dir') ~= 7
	        mkdir(save_dir);
	    end

	    try
	        exportgraphics(fig, save_path, 'Resolution', resolution);
	    catch
	        saveas(fig, save_path);
	    end
end


