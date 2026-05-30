function out = func_tr_sqp(scene, seed, criterion, sigma2, eps_s, reg_eps, tr_sqp)

	rng(seed + 1300);
	t0 = tic;

	M = size(scene.P0, 2);
	n_u = numel(scene.U0);
	x_init = [scene.P0(:); scene.U0(:)];
	lb = [scene.P_lb_each(:); -ones(n_u, 1)];
	ub = [scene.P_ub_each(:);  ones(n_u, 1)];

	opts = optimoptions('fmincon', ...
	    'Algorithm', 'sqp', ...
	    'Display', 'off', ...
	    'MaxIterations', tr_sqp.innerMaxIter, ...
	    'MaxFunctionEvaluations', tr_sqp.innerMaxFunctionEvaluations, ...
	    'StepTolerance', tr_sqp.stepTol, ...
	    'OptimalityTolerance', tr_sqp.optimalityTol, ...
	    'ConstraintTolerance', tr_sqp.constraintTol);

	x_cur = x_init;
	f_cur = func_objective(x_cur, scene.x0, M, criterion, sigma2, eps_s, reg_eps);
	x_best = x_cur;
	f_best = f_cur;
	Delta = tr_sqp.delta0;
	hist_f = nan(tr_sqp.maxIter + 1, 1);
	hist_f(1) = f_cur;
	n_hist = 1;

	for k = 1:tr_sqp.maxIter
	    x_trial = x_cur;
	    f_trial = f_cur;
	    try
	        [x_trial, f_trial] = fmincon( ...
	            @(x)func_objective(x, scene.x0, M, criterion, sigma2, eps_s, reg_eps), ...
	            x_cur, [], [], [], [], lb, ub, ...
	            @(x)func_trust_region_constraints(x, x_cur, Delta, M), opts);
	    catch
	    end

	    actual_drop = f_cur - f_trial;
	    step_norm = norm(x_trial - x_cur);
	    if isfinite(f_trial) && actual_drop > tr_sqp.acceptTol * max(1, abs(f_cur))
	        x_cur = x_trial;
	        f_cur = f_trial;
	        if f_cur < f_best
	            x_best = x_cur;
	            f_best = f_cur;
	        end

	        if step_norm > tr_sqp.eta2 * Delta
	            Delta = min(tr_sqp.deltaMax, tr_sqp.deltaGrow * Delta);
	        end
	    else
	        Delta = max(tr_sqp.deltaMin, tr_sqp.deltaShrink * Delta);
	    end

	    n_hist = n_hist + 1;
	    hist_f(n_hist) = f_best;
	    if step_norm < tr_sqp.stepTol || Delta <= tr_sqp.deltaMin
	        break;
	    end
	end
	hist_f = hist_f(1:n_hist);

	[P_final, U_final] = func_unpack(x_best, M);
	[J_final, min_sin] = func_fim(P_final, U_final, scene.x0, M, sigma2, eps_s);
	out = func_output('tr_sqp', 'TR-SQP', scene, P_final, U_final, hist_f, J_final, min_sin, toc(t0));
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

function [c, ceq] = func_trust_region_constraints(x, x_center, Delta, M)
	U = reshape(x((3 * M + 1):end), 3, M);
	c = norm(x - x_center)^2 - Delta^2;
	ceq = sum(U.^2, 1).' - 1;
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
	out.T = max(0, numel(hist_f) - 1);
	out.time_sec = time_sec;
end
