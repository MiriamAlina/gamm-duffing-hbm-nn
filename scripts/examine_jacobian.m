% examine_jacobian.m
% Compare Jacobians from:
%   A: hb_residual_aft  (FD Jacobian)
%   B: hb_residual_nn   (analytic/autodiff Jacobian)
%
% X layout expected:
%   X = [a0, a1, b1, a2, b2, ..., aH, bH, Om]  -> length = 2H+2
%
% Output per Om:
%   relJacErr_raw:        ||JA - JB|| / ||JA||
%   perm_row, alphaJ:     row permutation + global scaling to align JB to JA
%   relJacErr_perm_scale: ||JA - alpha*JB(perm,:)|| / ||JA||

clear; clc;

% --- USER SETTINGS -------------------------------------------------------
Om_list = linspace(1.08,1.16,10);

mu    = 1;
zeta  = 0.05;
kappa = 1;
gamma = 0.1;
P     = 0.18;

H = 3;
N = 4*H+1;
epsFD = 1e-8;
% ------------------------------------------------------------------------

% Initial guess (only 1st harmonic, linear FRF)
Om0 = Om_list(1);
Q1  = P / (kappa - mu*Om0^2 + 1i*zeta*Om0);

x0 = zeros(2*H+2,1);
x0(1)   = 0;          % a0
x0(2)   = real(Q1);   % a1
x0(3)   = -imag(Q1);  % b1 (your convention)
x0(end) = Om0;

rel_raw  = zeros(size(Om_list));
rel_fix  = zeros(size(Om_list));
perm_all = zeros(numel(Om_list), 2*H+1);
alpha_all= zeros(size(Om_list));

for k = 1:numel(Om_list)
    Om = Om_list(k);

    X = x0;
    X(end) = Om;

    % --- JA: FD Jacobian of AFT residual wrt ALL vars in X
    Smyopt = struct('epsrel',epsFD,'epsabs',epsFD,'ikeydx',1,'ikeyfd',1);
    JA = finite_difference_jacobian(@(Xin) hb_residual_aft(Xin,mu,zeta,kappa,gamma,P,H,N), X, Smyopt);

    % --- JB: analytic Jacobian from NN residual
    [~, JB] = hb_residual_nn(X,mu,zeta,kappa,gamma,P,H,N);

    % --- Align sizes: compare only w.r.t. coefficient variables (drop Om-column from JA)
    [mA,nA] = size(JA);
    [mB,nB] = size(JB);
    fprintf('Om = %.6f | size(JA)=%dx%d  size(JB)=%dx%d\n', Om, mA,nA,mB,nB);

    if mA ~= mB
        error('Row mismatch: residual size differs (mA=%d, mB=%d).', mA, mB);
    end

    if nA == nB + 1
        JA_cmp = JA(:,1:end-1);   % drop Om column
        JB_cmp = JB;
    elseif nA == nB
        JA_cmp = JA;
        JB_cmp = JB;
    else
        error('Column mismatch: cannot align (nA=%d, nB=%d).', nA, nB);
    end

    % --- Raw error (no permutation/scaling)
    rel_raw(k) = norm(JA_cmp - JB_cmp,'fro') / norm(JA_cmp,'fro');
    fprintf('           relJacErr_raw = %.3e\n', rel_raw(k));

    % --- Find row permutation by cosine similarity of Jacobian rows
    nEq = size(JA_cmp,1); % should be 2H+1
    S = zeros(nEq,nEq);
    for i = 1:nEq
        ai = JA_cmp(i,:).';
        for j = 1:nEq
            bj = JB_cmp(j,:).';
            S(i,j) = abs(ai' * bj) / (norm(ai)*norm(bj) + eps);
        end
    end

    perm_row = greedy_match_rows(S);          % JB row index for each JA row
    JBp = JB_cmp(perm_row,:);

    % --- Best global scaling alpha (least squares on all entries)
    alphaJ = (JA_cmp(:)'*JBp(:)) / (JBp(:)'*JBp(:) + eps);

    rel_fix(k) = norm(JA_cmp - alphaJ*JBp,'fro') / norm(JA_cmp,'fro');

    perm_all(k,:)  = perm_row;
    alpha_all(k)   = alphaJ;

    fprintf('           perm_row = [%s] | alpha = %.6g | relJacErr_fix = %.3e\n', ...
        sprintf('%d ', perm_row), alphaJ, rel_fix(k));

    RA = hb_residual_aft(X,mu,zeta,kappa,gamma,P,H,N);
    RB = hb_residual_nn(X,mu,zeta,kappa,gamma,P,H,N);
    relResErr = norm(RA - RB) / max(norm(RA), eps);
    fprintf('           relResErr = %.3e\n', relResErr);

end

fprintf('\nSummary over Om_list:\n');
fprintf('  mean relJacErr_raw = %.3e | max = %.3e\n', mean(rel_raw), max(rel_raw));
fprintf('  mean relJacErr_fix = %.3e | max = %.3e\n', mean(rel_fix), max(rel_fix));

% show typical permutation + scaling (median is robust)
perm_med  = round(median(perm_all,1));
alpha_med = median(alpha_all);
fprintf('\nTypical (median) alignment:\n');
fprintf('  perm_row_med = [%s]\n', sprintf('%d ', perm_med));
fprintf('  alpha_med    = %.6g\n', alpha_med);


% -------------------------------------------------------------------------
function perm = greedy_match_rows(S)
% Greedy matching: for each row i, pick the best unused column j
n = size(S,1);
perm = zeros(1,n);
used = false(1,n);
Swork = S;
for i = 1:n
    [~,j] = max(Swork(i,:));
    while used(j)
        Swork(i,j) = -Inf;
        [~,j] = max(Swork(i,:));
    end
    perm(i) = j;
    used(j) = true;
end
end

function J = finite_difference_jacobian(Hfuncname, x0, Smyopt, varargin)
for ii = 1:length(x0)

    switch Smyopt.ikeydx
        case 1
            dx = Smyopt.epsrel*abs(x0(ii));
            if dx == 0, dx = Smyopt.epsrel; end
        case 2
            dx = sqrt(Smyopt.epsabs)*(1+abs(x0(ii)));
        case 3
            dx = Smyopt.epsrel*abs(x0(ii));
            dx = max(dx,Smyopt.epsabs);
    end

    switch Smyopt.ikeyfd
        case 1 % forward
            if ii == 1
                f0 = feval(Hfuncname,x0,varargin{:});
                J  = zeros(length(f0),length(x0));
            end
            xp = x0; xp(ii) = x0(ii)+dx;
            dx = xp(ii)-x0(ii);
            fp = feval(Hfuncname,xp,varargin{:});
            J(:,ii) = (fp-f0)/dx;

        case 2 % backward
            if ii == 1
                f0 = feval(Hfuncname,x0,varargin{:});
                J  = zeros(length(f0),length(x0));
            end
            xm = x0; xm(ii) = x0(ii)-dx;
            dx = x0(ii)-xm(ii);
            fm = feval(Hfuncname,xm,varargin{:});
            J(:,ii) = (f0-fm)/dx;

        case 3 % central
            xp = x0; xp(ii) = x0(ii)+dx;
            xm = x0; xm(ii) = x0(ii)-dx;
            dx = (xp(ii)-xm(ii))/2;
            fp = feval(Hfuncname,xp,varargin{:});
            fm = feval(Hfuncname,xm,varargin{:});
            if ii == 1
                J = zeros(length(fp),length(x0));
            end
            J(:,ii) = (fp-fm)/(2*dx);
    end
end
end