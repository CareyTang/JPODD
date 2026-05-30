function [g, aux] = func_calc_jacobian(p, u, x0, eps_s)

	if nargin < 4 || isempty(eps_s)
	    eps_s = 0;
	end

	p = p(:);
	u = u(:);
	x0 = x0(:);

	nu = norm(u);
	if nu < 1e-12
	    u = [1; 0; 0];
	else
	    u = u / nu;
	end

	r = x0 - p;
	d = norm(r);

	if d < 1e-12
	    e = [0; 0; 1];
	    Pperp = eye(3);
	    mu = 0;
	    sin_theta = 1;
	    g = zeros(3, 1);
	    v = [1; 0; 0];
	else
	    e = r / d;
	    Pperp = eye(3) - e * e.';
	    mu = max(-1, min(1, u.' * e));
	    sin_theta = sqrt(max(0, 1 - mu^2));

	    proj_u = Pperp * u;
	    proj_norm = norm(proj_u);
	    if proj_norm < 1e-12
	        v = zeros(3, 1);
	    else
	        v = proj_u / proj_norm;
	    end

	    g = proj_u / (d * max(sin_theta, 1e-12));
	end

	aux = struct();
	aux.d = d;
	aux.e = e;
	aux.Pperp = Pperp;
	aux.mu = mu;
	aux.sin_theta = sin_theta;
	aux.is_nondegenerate = sin_theta >= eps_s;
	aux.u = u;
	aux.v = v;
	aux.g = g;
end


