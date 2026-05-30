function out = func_de(scene, seed, criterion, sigma2, eps_s, reg_eps, de)
	rng(seed + 1200);
	t0 = tic;

	M = size(scene.P0, 2);
	n_p = numel(scene.P0);
	n_u = numel(scene.U0);
	n_x = n_p + n_u;

	lb = [scene.P_lb_each(:); -ones(n_u, 1)];
	ub = [scene.P_ub_each(:);  ones(n_u, 1)];
	x0 = [scene.P0(:); scene.U0(:)];

	pop_size = de.populationSize;
	max_iter = de.maxIter;
	F = de.mutationFactor;
	CR = de.crossoverRate;

	X = repmat(lb.', pop_size, 1) + ...
	    rand(pop_size, n_x) .* repmat((ub - lb).', pop_size, 1);
	X(1, :) = x0.';

	fit = inf(pop_size, 1);
	for i = 1:pop_size
	    fit(i) = func_objective(X(i, :).', scene.x0, M, criterion, sigma2, eps_s, reg_eps);
	end

	[best_f, best_idx] = min(fit);
	best_x = X(best_idx, :);
	hist_f = nan(max_iter + 1, 1);
	hist_f(1) = best_f;

	for k = 1:max_iter
	    for i = 1:pop_size
	        idx = randperm(pop_size, 3);
	        while any(idx == i)
	            idx = randperm(pop_size, 3);
	        end

	        mutant = X(idx(1), :) + F * (X(idx(2), :) - X(idx(3), :));
	        mutant = min(max(mutant, lb.'), ub.');

	        cross = rand(1, n_x) < CR;
	        cross(randi(n_x)) = true;
	        trial = X(i, :);
	        trial(cross) = mutant(cross);

	        trial_f = func_objective(trial.', scene.x0, M, criterion, sigma2, eps_s, reg_eps);
	        if trial_f < fit(i)
	            X(i, :) = trial;
	            fit(i) = trial_f;
	        end
	    end

	    [trial_best, best_idx] = min(fit);
	    if trial_best < best_f
	        best_f = trial_best;
	        best_x = X(best_idx, :);
	    end
	    hist_f(k + 1) = best_f;
	end

	[P_final, U_final] = func_unpack(best_x.', M);
	[J_final, min_sin] = func_fim(P_final, U_final, scene.x0, M, sigma2, eps_s);
	out = func_output('de', 'DE', scene, P_final, U_final, hist_f, J_final, min_sin, toc(t0));
end

function f = func_objective(x, x0, M, criterion, sigma2, eps_s, reg_eps)
	[P, U] = func_unpack(x, M);
	[J, min_sin] = func_fim(P, U, x0, M, sigma2, eps_s);
	if min_sin < eps_s || ~all(isfinite(J(:)))
	    f = 1e30;
	    return;
	end

	J = (J + J.') / 2 + reg_eps * eye(3);
	if strcmpi(criterion, 'aopt')
	    f = trace(inv(J));
	else
	    detJ = det(J);
	    if detJ <= 0 || ~isfinite(detJ)
	        f = 1e30;
	    else
	        f = -log(detJ);
	    end
	end
end

function [P, U] = func_unpack(x, M)
	P = reshape(x(1:(3 * M)), 3, M);
	U = reshape(x((3 * M + 1):end), 3, M);
	for i = 1:M
	    n = norm(U(:, i));
	    if n < 1e-12
	        U(:, i) = [1; 0; 0];
	    else
	        U(:, i) = U(:, i) / n;
	    end
	end
end

function [J, min_sin] = func_fim(P, U, x0, M, sigma2, eps_s)
	J = zeros(3, 3);
	min_sin = inf;
	for i = 1:M
	    sigma2_i = func_sigma2(sigma2, i);
	    [g_i, aux_i] = func_calc_jacobian(P(:, i), U(:, i), x0, eps_s);
	    J = J + (g_i * g_i.') / sigma2_i;
	    min_sin = min(min_sin, aux_i.sin_theta);
	end
	J = (J + J.') / 2;
end

function s = func_sigma2(sigma2, i)
	if isscalar(sigma2)
	    s = sigma2;
	else
	    s = sigma2(i);
	end
end

function out = func_output(method_id, method_name, scene, P, U, hist_f, J, min_sin, time_sec)
	out = struct();
	out.method_id = method_id;
	out.method_name = method_name;
	out.scene_name = scene.scene_name;
	out.P_final = P;
	out.U_final = U;
	out.P_init = scene.P0;
	out.U_init = scene.U0;
	out.P_hist = cat(3, scene.P0, P);
	out.U_hist = cat(3, scene.U0, U);
	out.x0 = scene.x0;
	out.lb = scene.lb;
	out.ub = scene.ub;
	out.P_lb_each = scene.P_lb_each;
	out.P_ub_each = scene.P_ub_each;
	out.hist_f = hist_f(:);
	out.hist_min_eigJ = repmat(min(eig(J)), numel(hist_f), 1);
	out.hist_min_sin = repmat(min_sin, numel(hist_f), 1);
	out.T = numel(hist_f) - 1;
	out.time_sec = time_sec;
end
