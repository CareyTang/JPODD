function out = func_manopt(scene, seed, criterion, sigma2, eps_s, reg_eps, manopt)

	if exist('productmanifold', 'file') ~= 2 || exist('obliquefactory', 'file') ~= 2
	    error('Manopt is required for func_manopt. Add Manopt to the MATLAB path before running this baseline.');
	end

	old_warning_state = warning('query', 'manopt:getHessian:approx');
	warning('off', 'manopt:getHessian:approx');
	cleanup_warning = onCleanup(@() warning(old_warning_state.state, 'manopt:getHessian:approx'));

	rng(seed + 1400);
	t0 = tic;

	M = size(scene.P0, 2);
	n_p = numel(scene.P0);
	p_lb = scene.P_lb_each(:);
	p_ub = scene.P_ub_each(:);
	p_mid = 0.5 * (p_lb + p_ub);
	p_rad = 0.5 * max(p_ub - p_lb, 1e-12);
	z0 = atanh(max(-0.999, min(0.999, (scene.P0(:) - p_mid) ./ p_rad)));

	manifold = productmanifold(struct( ...
	    'z', euclideanfactory(n_p), ...
	    'U', obliquefactory(3, M)));

	problem.M = manifold;
	problem.cost = @(X)func_objective_from_manopt( ...
	    X, p_mid, p_rad, scene.x0, M, criterion, sigma2, eps_s, reg_eps);
	problem.egrad = @(X)func_egrad( ...
	    X, p_mid, p_rad, scene.x0, M, criterion, sigma2, eps_s, reg_eps, manopt.fdEps);

	X0 = struct();
	X0.z = z0;
	X0.U = scene.U0;

	options.maxiter = manopt.maxIter;
	options.verbosity = 0;
	options.tolgradnorm = manopt.tolGradNorm;

	try
	    [X_best, f_best, info] = trustregions(problem, X0, options);
	catch
	    [X_best, f_best, info] = steepestdescent(problem, X0, options);
	end

	hist_f = func_manopt_history(info, f_best);
	P_final = reshape(p_mid + p_rad .* tanh(X_best.z), 3, M);
	U_final = X_best.U;
	[J_final, min_sin] = func_fim(P_final, U_final, scene.x0, M, sigma2, eps_s);
	out = func_output('manopt', 'Manopt', scene, P_final, U_final, hist_f, J_final, min_sin, toc(t0));
end

function f = func_objective_from_manopt(X, p_mid, p_rad, x0, M, criterion, sigma2, eps_s, reg_eps)
	P = reshape(p_mid + p_rad .* tanh(X.z), 3, M);
	U = X.U;
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

function egrad = func_egrad(X, p_mid, p_rad, x0, M, criterion, sigma2, eps_s, reg_eps, eps_fd)
	egrad = struct();
	egrad.z = zeros(size(X.z));
	egrad.U = zeros(size(X.U));

	for k = 1:numel(X.z)
	    h = eps_fd * max(1, abs(X.z(k)));
	    Xp = X;
	    Xm = X;
	    Xp.z(k) = Xp.z(k) + h;
	    Xm.z(k) = Xm.z(k) - h;
	    fp = func_objective_from_manopt(Xp, p_mid, p_rad, x0, M, criterion, sigma2, eps_s, reg_eps);
	    fm = func_objective_from_manopt(Xm, p_mid, p_rad, x0, M, criterion, sigma2, eps_s, reg_eps);
	    egrad.z(k) = (fp - fm) / (2 * h);
	end

	for k = 1:numel(X.U)
	    h = eps_fd * max(1, abs(X.U(k)));
	    Xp = X;
	    Xm = X;
	    Xp.U(k) = Xp.U(k) + h;
	    Xm.U(k) = Xm.U(k) - h;
	    fp = func_objective_from_manopt(Xp, p_mid, p_rad, x0, M, criterion, sigma2, eps_s, reg_eps);
	    fm = func_objective_from_manopt(Xm, p_mid, p_rad, x0, M, criterion, sigma2, eps_s, reg_eps);
	    egrad.U(k) = (fp - fm) / (2 * h);
	end
end

function hist_f = func_manopt_history(info, f_best)
	if isstruct(info) && isfield(info, 'cost')
	    hist_f = [info.cost].';
	else
	    hist_f = f_best;
	end
	if isempty(hist_f)
	    hist_f = f_best;
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
