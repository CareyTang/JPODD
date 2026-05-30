function scene = func_init_free(seed, M, sigma2, eps_s, reg_eps, initMaxTry, big_center, ...
    big_size, target_offset_z, target_size, beta_max_deg)
	rng(seed);

	lb = big_center(:) - big_size(:) / 2;
	ub = big_center(:) + big_size(:) / 2;
	P_lb_each = repmat(lb(:), 1, M);
	P_ub_each = repmat(ub(:), 1, M);

	target_center = big_center(:) + [0; 0; target_offset_z];
	x_lb = target_center - target_size(:) / 2;
	x_ub = target_center + target_size(:) / 2;
	x0 = func_rand_in_box(x_lb, x_ub, 1);

	ok = false;
	P0 = [];
	U0 = [];
	for trial = 1:initMaxTry
	    P_try = func_rand_in_boxes(P_lb_each, P_ub_each);
	    U_try = func_rand_transverse_units(P_try, x0, beta_max_deg);
	    [J_try, sensor_try] = func_total_fim(P_try, U_try, x0, M, sigma2, eps_s);
	    min_sin = min([sensor_try.sin_theta]);
	    min_eig = min(eig((J_try + J_try.') / 2));

	    if isfinite(min_sin) && isfinite(min_eig) && ...
	            min_sin >= eps_s && min_eig > 10 * reg_eps
	        P0 = P_try;
	        U0 = U_try;
	        ok = true;
	        break;
	    end
	end

	if ~ok
	    P0 = func_rand_in_boxes(P_lb_each, P_ub_each);
	    U0 = func_rand_transverse_units(P0, x0, beta_max_deg);
	end

	scene = struct();
	scene.scene_name = 'free_space';
	scene.vertex_centers = [];
	scene.x0 = x0;
	scene.P0 = P0;
	scene.U0 = func_normalize_cols(U0);
	scene.lb = min(P_lb_each, [], 2);
	scene.ub = max(P_ub_each, [], 2);
	scene.P_lb_each = P_lb_each;
	scene.P_ub_each = P_ub_each;
	scene.x_lb = x_lb;
	scene.x_ub = x_ub;
end

function X = func_rand_in_box(lb, ub, M)
	X = lb + (ub - lb) .* rand(3, M);
	if M == 1
	    X = X(:);
	end
end

function X = func_rand_in_boxes(lb_each, ub_each)
	X = lb_each + (ub_each - lb_each) .* rand(size(lb_each));
end

function [J, sensorData] = func_total_fim(P, U, x0, M, sigma2, eps_s)
	J = zeros(3, 3);
	sensorData = repmat(func_sensor_template(), 1, M);
	for i = 1:M
	    sigma2_i = func_sigma2(sigma2, i);
	    [g_i, aux_i] = func_calc_jacobian(P(:, i), U(:, i), x0, eps_s);
	    aux_i.g = g_i;
	    aux_i.Ji = (g_i * g_i.') / sigma2_i;
	    sensorData(i) = aux_i;
	    J = J + aux_i.Ji;
	end
	J = (J + J.') / 2;
end

function U = func_rand_transverse_units(P, x0, beta_max_deg)
	M = size(P, 2);
	U = zeros(3, M);
	s_beta = func_tilt_limit_sine(beta_max_deg);

	for i = 1:M
	    r = x0 - P(:, i);
	    d = norm(r);
	    if d < 1e-12
	        U(:, i) = [1; 0; 0];
	        continue;
	    end

	    e = r / d;
	    v = zeros(3, 1);
	    for k = 1:20
	        w = randn(3, 1);
	        v = w - e * (e.' * w);
	        if norm(v) > 1e-10
	            break;
	        end
	    end

	    if norm(v) <= 1e-10
	        basis = func_plane_basis(e);
	        v = basis(:, 1);
	    end

	    if isfinite(s_beta) && s_beta < 1 - 1e-12
	        v = func_enforce_tilt_limit(v, e, s_beta);
	    end

	    U(:, i) = v / norm(v);
	end
end

function U = func_normalize_cols(U)
	for i = 1:size(U, 2)
	    ni = norm(U(:, i));
	    if ni < 1e-12
	        U(:, i) = [1; 0; 0];
	    else
	        U(:, i) = U(:, i) / ni;
	    end
	end
end

function v = func_enforce_tilt_limit(v, e, s_beta)
	v = v(:);
	if abs(v(3)) <= s_beta + 1e-10
	    return;
	end

	basis = func_plane_basis(e);
	c = basis(3, :).';
	rho = norm(c);
	if rho <= max(s_beta, 1e-10)
	    return;
	end

	v_horizontal = func_horizontal_transverse_unit(e);
	z_horizontal = basis.' * v_horizontal;
	if norm(z_horizontal) > 1e-12
	    z_horizontal = z_horizontal / norm(z_horizontal);
	    v = basis * z_horizontal;
	else
	    c_hat = c / rho;
	    c_perp = [-c_hat(2); c_hat(1)];
	    v = basis * c_perp;
	end
end

function v = func_horizontal_transverse_unit(e)
	h = [-e(2); e(1); 0];
	if norm(h) <= 1e-12
	    v = [1; 0; 0];
	else
	    v = h / norm(h);
	end
end

function s_beta = func_tilt_limit_sine(beta_max_deg)
	beta_deg = min(max(real(beta_max_deg), 0), 90);
	s_beta = sind(beta_deg);
end

function basis = func_plane_basis(e)
	[~, idx] = min(abs(e));
	I3 = eye(3);
	a = I3(:, idx);
	b1 = a - e * (e.' * a);
	if norm(b1) < 1e-12
	    a = I3(:, mod(idx, 3) + 1);
	    b1 = a - e * (e.' * a);
	end
	b1 = b1 / max(norm(b1), 1e-12);
	b2 = cross(e, b1);
	b2 = b2 / max(norm(b2), 1e-12);
	basis = [b1, b2];
end

function s = func_sigma2(sigma2, i)
	if isscalar(sigma2)
	    s = sigma2;
	else
	    s = sigma2(i);
	end
end

function tmpl = func_sensor_template()
	tmpl = struct( ...
	    'd', NaN, ...
	    'e', zeros(3, 1), ...
	    'Pperp', zeros(3, 3), ...
	    'mu', NaN, ...
	    'sin_theta', NaN, ...
	    'is_nondegenerate', false, ...
	    'u', zeros(3, 1), ...
	    'v', zeros(3, 1), ...
	    'g', zeros(3, 1), ...
	    'Ji', zeros(3, 3));
end
