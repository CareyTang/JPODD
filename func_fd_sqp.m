function out = func_fd_sqp(scene, criterion, sigma2, eps_s, reg_eps, maxIter, ...
    stop_rel, beta_max_deg, generic_sqp)

	P = scene.P0;
	U = scene.U0;
	x0 = scene.x0;
	P_init = P;
	U_init = U;
	M = size(P, 2);

	[J_total, sensorData] = func_total_fim(P, U, x0, sigma2, eps_s);
	P_hist = zeros(3, M, maxIter + 1);
	U_hist = zeros(3, M, maxIter + 1);
	hist_f = nan(maxIter + 1, 1);
	hist_min_eigJ = nan(maxIter + 1, 1);
	hist_min_sin = nan(maxIter + 1, 1);
	hist_block_iter = nan(maxIter, M);
	hist_exitflag = nan(maxIter, M);
	hist_model_attempt = nan(maxIter, M);
	hist_step_norm = nan(maxIter, M);
	hist_tau = nan(maxIter, M);

	state = repmat(struct( ...
	    'tau', generic_sqp.tau0, ...
	    'qp_step', zeros(3, 1)), 1, M);

	P_hist(:, :, 1) = P;
	U_hist(:, :, 1) = U;
	hist_f(1) = func_objective_from_J(J_total, criterion, reg_eps);
	hist_min_eigJ(1) = min(eig((J_total + J_total.') / 2));
	hist_min_sin(1) = min([sensorData.sin_theta]);

	t0 = tic;
	n_sweeps = 0;
	for t = 1:maxIter
	    n_sweeps = t;
	    for j = 1:M
	        sigma2_j = func_sigma2(sigma2, j);
	        S_true = (J_total - sensorData(j).Ji + (J_total - sensorData(j).Ji).') / 2;
	        p_old = P(:, j);
	        u_old = U(:, j);
	        lb = scene.P_lb_each(:, j);
	        ub = scene.P_ub_each(:, j);

	        [p_new, u_new, info, state(j)] = func_update_sensor( ...
	            p_old, u_old, S_true, lb, ub, x0, sigma2_j, criterion, ...
	            eps_s, reg_eps, beta_max_deg, generic_sqp, state(j));
	        P(:, j) = p_new;
	        U(:, j) = u_new;

	        [g_new, aux_new] = func_calc_jacobian(P(:, j), U(:, j), x0, eps_s);
	        Ji_new = (g_new * g_new.') / sigma2_j;
	        aux_new.Ji = Ji_new;
	        sensorData(j) = aux_new;
	        J_total = (S_true + Ji_new + S_true.' + Ji_new.') / 2;

	        hist_block_iter(t, j) = info.iter_used;
	        hist_exitflag(t, j) = info.exitflag;
	        hist_model_attempt(t, j) = info.model_attempt;
	        hist_step_norm(t, j) = info.step_norm;
	        hist_tau(t, j) = info.tau;
	    end

	    [J_total, sensorData] = func_total_fim(P, U, x0, sigma2, eps_s);
	    P_hist(:, :, t + 1) = P;
	    U_hist(:, :, t + 1) = U;
	    hist_f(t + 1) = func_objective_from_J(J_total, criterion, reg_eps);
	    hist_min_eigJ(t + 1) = min(eig((J_total + J_total.') / 2));
	    hist_min_sin(t + 1) = min([sensorData.sin_theta]);

	    if abs(hist_f(t + 1) - hist_f(t)) / max(1e-12, abs(hist_f(t))) < stop_rel
	        break;
	    end
	end

	out = func_output('fd_sqp', 'FD-SQP', scene, P, U, P_init, U_init, P_hist, U_hist, ...
	    hist_f, hist_min_eigJ, hist_min_sin, hist_block_iter, hist_exitflag, ...
	    hist_model_attempt, hist_step_norm, n_sweeps, toc(t0));
	out.hist_tau = hist_tau(1:n_sweeps, :);
end

function [p_new, u_new, info, state] = func_update_sensor(p, u, S_true, lb, ub, ...
    x0, sigma2_i, criterion, eps_s, reg_eps, beta_max_deg, generic_sqp, state)
	info = func_info();
	info.tau = state.tau;
	[phi, grad, ~, u_base, ok] = func_reduced_eval( ...
	    p, u, S_true, x0, sigma2_i, criterion, eps_s, reg_eps, beta_max_deg);
	p_new = p;
	u_new = u_base;
	if ~ok || norm(grad) <= generic_sqp.stepTol
	    return;
	end

	H0 = func_fd_hessian(p, u_base, S_true, x0, sigma2_i, criterion, ...
	    eps_s, reg_eps, beta_max_deg, generic_sqp, grad);
	tau = max(state.tau, generic_sqp.tau0);
	step0 = func_project_step(state.qp_step, lb - p, ub - p);
	for attempt = 1:generic_sqp.qpMaxAttempts
	    [H, tau_used] = func_regularize(H0, tau, generic_sqp);
	    [step, exitflag, iter_used] = func_quadprog(H, grad, lb - p, ub - p, step0, generic_sqp);
	    info.tau = tau_used;
	    info.exitflag = exitflag;
	    info.iter_used = iter_used;
	    info.model_attempt = attempt;
	    if ~all(isfinite(step))
	        tau = func_next_tau(tau_used, generic_sqp);
	        step0 = zeros(3, 1);
	        continue;
	    end

	    info.step_norm = norm(step);
	    p_try = p + step;
	    [phi_try, ~, ~, u_try, ok_try] = func_reduced_eval( ...
	        p_try, u_base, S_true, x0, sigma2_i, criterion, eps_s, reg_eps, beta_max_deg);
	    if ok_try && phi_try <= phi + generic_sqp.acceptTol * max(1, abs(phi))
	        p_new = p_try;
	        u_new = u_try;
	        state.tau = tau_used;
	        state.qp_step = step;
	        return;
	    end
	    step0 = func_project_step(step, lb - p, ub - p);
	    tau = func_next_tau(tau_used, generic_sqp);
	end
	state.tau = min(generic_sqp.tauMax, tau);
	state.qp_step = zeros(3, 1);
end

function H = func_fd_hessian(p, u, S_true, x0, sigma2_i, criterion, eps_s, ...
    reg_eps, beta_max_deg, generic_sqp, g0)
	H = zeros(3, 3);
	for k = 1:3
	    h = generic_sqp.hess_fd_eps * max(1, abs(p(k)));
	    e = zeros(3, 1);
	    e(k) = h;
	    [~, gp, ~, ~, okp] = func_reduced_eval(p + e, u, S_true, x0, sigma2_i, criterion, eps_s, reg_eps, beta_max_deg);
	    [~, gm, ~, ~, okm] = func_reduced_eval(p - e, u, S_true, x0, sigma2_i, criterion, eps_s, reg_eps, beta_max_deg);
	    if okp && okm
	        H(:, k) = (gp - gm) / (2 * h);
	    elseif okp
	        H(:, k) = (gp - g0) / h;
	    elseif okm
	        H(:, k) = (g0 - gm) / h;
	    end
	end
	H = (H + H.') / 2;
end

function [J, sensorData] = func_total_fim(P, U, x0, sigma2, eps_s)
	M = size(P, 2);
	J = zeros(3, 3);
	sensorData = repmat(func_sensor_template(), 1, M);
	for i = 1:M
	    sigma2_i = func_sigma2(sigma2, i);
	    [g_i, aux_i] = func_calc_jacobian(P(:, i), U(:, i), x0, eps_s);
	    aux_i.Ji = (g_i * g_i.') / sigma2_i;
	    sensorData(i) = aux_i;
	    J = J + aux_i.Ji;
	end
	J = (J + J.') / 2;
end

function [phi, grad, model, u_star, ok] = func_reduced_eval(p, u, S_true, x0, sigma2_i, criterion, eps_s, reg_eps, beta_max_deg)
	u_star = func_update_orientation(p, u, S_true + reg_eps * eye(3), x0, sigma2_i, criterion, beta_max_deg);
	[g, aux] = func_calc_jacobian(p, u_star, x0, eps_s);
	Ji = (g * g.') / sigma2_i;
	J = S_true + Ji;
	phi = func_objective_from_J(J, criterion, reg_eps);
	[grad, model] = func_gradient_model(p, u_star, S_true, x0, sigma2_i, criterion, eps_s, reg_eps);
	ok = isfinite(phi) && aux.sin_theta >= eps_s && all(isfinite(grad));
end

function [grad, model] = func_gradient_model(p, u, S_true, x0, sigma2_i, criterion, eps_s, reg_eps)
	[g, aux] = func_calc_jacobian(p, u, x0, eps_s);
	Ji = (g * g.') / sigma2_i;
	d = aux.d;
	e = aux.e;
	v = aux.v;
	c = max(-1, min(1, u.' * e));
	s = max(aux.sin_theta, 1e-12);
	lambda = 1 / (sigma2_i * d^2);
	Hgeo = (e * v.' + (c / s) * (eye(3) - e * e.' - v * v.')) / d;
	J = S_true + Ji + reg_eps * eye(3);
	Jinv = inv((J + J.') / 2);
	if strcmpi(criterion, 'aopt')
	    W = Jinv * Jinv;
	else
	    W = Jinv;
	end
	grad = -(2 * lambda / d) * real(v.' * W * v) * e - 2 * lambda * (Hgeo.' * (W * v));
	model = struct('lambda', lambda, 'H', Hgeo, 'W', (W + W.') / 2, 'Pr', e * e.', 'Pt', eye(3) - e * e.');
end

function u = func_update_orientation(p, u_ref, S_eval, x0, sigma2_i, criterion, beta_max_deg)
	[~, aux] = func_calc_jacobian(p, u_ref, x0, 0);
	basis = func_plane_basis(aux.e);
	Sinv_basis = S_eval \ basis;
	if strcmpi(criterion, 'aopt')
	    a = 1 / (sigma2_i * aux.d^2);
	    A2 = a * (Sinv_basis.' * Sinv_basis);
	    B2 = eye(2) + a * (basis.' * Sinv_basis);
	    [V, D] = eig((A2 + A2.') / 2, (B2 + B2.') / 2);
	    [~, idx] = max(real(diag(D)));
	else
	    A2 = basis.' * Sinv_basis;
	    [V, D] = eig((A2 + A2.') / 2);
	    [~, idx] = max(real(diag(D)));
	end
	u = basis * real(V(:, idx));
	u = u / norm(u);
	if dot(u, aux.Pperp * aux.u) < 0
	    u = -u;
	end
	u = func_enforce_tilt(u, aux.e, beta_max_deg);
end

function u = func_enforce_tilt(u, e, beta_max_deg)
	s_beta = sind(beta_max_deg);
	if s_beta >= 1
	    return;
	end
	h = [u(1); u(2)];
	if norm(h) > s_beta
	    h = h / norm(h) * s_beta;
	    z = sqrt(max(0, 1 - h.' * h));
	    u = [h; sign(u(3) + (u(3) == 0)) * z];
	    u = u - e * (e.' * u);
	    u = u / norm(u);
	end
end

function [H, tau] = func_regularize(H0, tau, prm)
	H0 = (H0 + H0.') / 2;
	tau = max(tau, prm.tau0);
	for k = 1:20
	    H = H0 + tau * eye(3);
	    if min(eig((H + H.') / 2)) >= prm.minModelEig
	        return;
	    end
	    tau = min(prm.tauMax, prm.tauScale * tau);
	end
	H = H0 + tau * eye(3);
end

function [step, exitflag, iter_used] = func_quadprog(H, g, lb, ub, x0, prm)
	opts = optimoptions('quadprog', 'Algorithm', prm.qpAlgorithm, 'Display', 'off', ...
	    'MaxIterations', prm.qpMaxIterations, 'OptimalityTolerance', prm.qpOptimalityTolerance, ...
	    'ConstraintTolerance', prm.qpConstraintTolerance);
	try
	    [step, ~, exitflag, output] = quadprog(H, g, [], [], [], [], lb, ub, x0, opts);
	    iter_used = output.iterations;
	catch
	    step = nan(3, 1);
	    exitflag = 0;
	    iter_used = 0;
	end
end

function step = func_project_step(step, lb, ub)
	step = min(max(step(:), lb), ub);
end

function tau = func_next_tau(tau, prm)
	tau = min(prm.tauMax, prm.tauScale * max([tau, prm.tau0, prm.minModelEig]));
end

function f = func_objective_from_J(J, criterion, reg_eps)
	J = (J + J.') / 2;
	if strcmpi(criterion, 'aopt')
	    f = trace(inv(J + reg_eps * eye(3)));
	else
	    f = -log(det(J + reg_eps * eye(3)));
	end
end

function s = func_sigma2(sigma2, i)
	if isscalar(sigma2)
	    s = sigma2;
	else
	    s = sigma2(i);
	end
end

function info = func_info()
	info = struct('eta_r', NaN, 'eta_t', NaN, 'tau', NaN, 'iter_used', 0, ...
	    'exitflag', 0, 'model_attempt', 0, 'step_norm', 0);
end

function basis = func_plane_basis(e)
	[~, idx] = min(abs(e));
	a = eye(3);
	b1 = a(:, idx) - e * (e.' * a(:, idx));
	if norm(b1) < 1e-12
	    b1 = a(:, mod(idx, 3) + 1) - e * (e.' * a(:, mod(idx, 3) + 1));
	end
	b1 = b1 / norm(b1);
	b2 = cross(e, b1);
	b2 = b2 / norm(b2);
	basis = [b1, b2];
end

function tmpl = func_sensor_template()
	tmpl = struct('d', NaN, 'e', zeros(3, 1), 'Pperp', zeros(3, 3), ...
	    'mu', NaN, 'sin_theta', NaN, 'is_nondegenerate', false, ...
	    'u', zeros(3, 1), 'v', zeros(3, 1), 'g', zeros(3, 1), 'Ji', zeros(3, 3));
end

function out = func_output(method_id, method_name, scene, P, U, P_init, U_init, ...
    P_hist, U_hist, hist_f, hist_min_eigJ, hist_min_sin, hist_block_iter, ...
    hist_exitflag, hist_model_attempt, hist_step_norm, n_sweeps, time_sec)
	out = struct();
	out.method_id = method_id;
	out.method_name = method_name;
	out.scene_name = scene.scene_name;
	out.P_final = P;
	out.U_final = U;
	out.P_init = P_init;
	out.U_init = U_init;
	out.P_hist = P_hist(:, :, 1:(n_sweeps + 1));
	out.U_hist = U_hist(:, :, 1:(n_sweeps + 1));
	out.x0 = scene.x0;
	out.lb = scene.lb;
	out.ub = scene.ub;
	out.P_lb_each = scene.P_lb_each;
	out.P_ub_each = scene.P_ub_each;
	out.x_lb = scene.x_lb;
	out.x_ub = scene.x_ub;
	out.hist_f = hist_f(1:(n_sweeps + 1));
	out.hist_min_eigJ = hist_min_eigJ(1:(n_sweeps + 1));
	out.hist_min_sin = hist_min_sin(1:(n_sweeps + 1));
	out.hist_block_iter = hist_block_iter(1:n_sweeps, :);
	out.hist_exitflag = hist_exitflag(1:n_sweeps, :);
	out.hist_model_attempt = hist_model_attempt(1:n_sweeps, :);
	out.hist_step_norm = hist_step_norm(1:n_sweeps, :);
	out.T = n_sweeps;
	out.time_sec = time_sec;
end
