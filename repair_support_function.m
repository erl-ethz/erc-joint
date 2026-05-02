function [r_new, dr_new, ddr_new] = repair_support_function(x, r, dr, ddr, eps_val)

    x   = x(:);
    r   = r(:);
    dr  = dr(:);
    ddr = ddr(:);
    n  = length(x);
    dx = x(2) - x(1);

    % Compute curvature and clamp
    rho = r + ddr;
    g   = max(rho, eps_val);

    % Build operator (r'' + r)
    e  = ones(n,1);
    D2 = spdiags([e -2*e e], -1:1, n, n) / dx^2;
    A  = D2 + speye(n);

    % Impose C1 boundary conditions

    % Left value
    A(1,:) = 0;
    A(1,1) = 1;
    g(1)   = r(1);

    % Left slope
    A(2,:) = 0;
    A(2,1:3) = [-3 4 -1] / (2*dx);
    g(2)   = dr(1);

    % Right slope
    A(end-1,:) = 0;
    A(end-1,end-2:end) = [1 -4 3] / (2*dx);
    g(end-1) = dr(end);

    % Right value
    A(end,:) = 0;
    A(end,end) = 1;
    g(end) = r(end);

    % Solve system
    r_new = A \ g;

    % Compute derivatives
    dr_new  = zeros(n,1);
    ddr_new = zeros(n,1);

    % Interior
    dr_new(2:end-1)  = (r_new(3:end) - r_new(1:end-2)) / (2*dx);
    ddr_new(2:end-1) = (r_new(3:end) - 2*r_new(2:end-1) + r_new(1:end-2)) / dx^2;

    % Enforce boundary slopes
    dr_new(1)   = dr(1);
    dr_new(end) = dr(end);

    % Boundary second derivative
    ddr_new(1)       = ddr_new(2);
    ddr_new(end-1)   = ddr_new(end-2);
    ddr_new(end)     = ddr_new(end-2);

    r_new   = r_new';
    dr_new  = dr_new';
    ddr_new = ddr_new';
    
end