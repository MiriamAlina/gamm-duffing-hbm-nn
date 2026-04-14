%========================================================================
% DESCRIPTION: 
% Matlab function setting up the frequency-domain residual vector 'R' and 
% its derivatives for given frequency 'Om' and vector of harmonics of the 
% generalized coordiantes 'Q'. The corresponding model is a single DOF
% oscillator with cubic spring nonlinearity, i.e. the Duffing oscillator,
% governed by the time-domain equation of motion
% 
%       mu * \ddot q + zeta * \dot q + kappa * q + gamma * q^3 = P * cos(Om*t).
% 
% Compared to the more general variant 'HB_residual', we
%       - consider only a single-DOF oscillator, instead of multi-DOF
%       systems
%       - consider only a cubic spring nonlinearity, instead of various
%       nonlinear elements
%       - carry out only frequency response analysis, instead of e.g. 
%       nonlinear modal analysis, and
%       - do not calculate analytical derivatives.
%========================================================================
% This file is part of NLvib.
% 
% If you use NLvib, please refer to the book:
%   M. Krack, J. Gross: Harmonic Balance for Nonlinear Vibration
%   Problems. Springer, 2019. https://doi.org/10.1007/978-3-030-14023-6.
% 
% COPYRIGHT AND LICENSING: 
% NLvib Version 1.3 Copyright (C) 2020  Malte Krack  
%										(malte.krack@ila.uni-stuttgart.de) 
%                     					Johann Gross 
%										(johann.gross@ila.uni-stuttgart.de)
%                     					University of Stuttgart
% This program comes with ABSOLUTELY NO WARRANTY. 
% NLvib is free software, you can redistribute and/or modify it under the
% GNU General Public License as published by the Free Software Foundation,
% either version 3 of the License, or (at your option) any later version.
% For details on license and warranty, see http://www.gnu.org/licenses
% or gpl-3.0.txt.
%========================================================================
% Adjustments as part of the AICIM project by Miriam Goldack
%========================================================================
function [R, dR, Q] = hb_residual_nn(X,mu,zeta,kappa,gamma,P,H,N)

persistent pyModule1 pyModule2

mFileFolder = fileparts(mfilename('fullpath'));
[~, folderName] = fileparts(mFileFolder);  % check whether the current folder is 'scripts'
if ~strcmp(folderName, 'scripts')
    pyFolder = fullfile(mFileFolder, 'scripts');
else
    pyFolder = mFileFolder;
end
if count(py.sys.path, pyFolder) == 0
    insert(py.sys.path, int32(0), pyFolder);  % add the current folder to Python sys.path 
end

if isempty(pyModule1) || isempty(pyModule2)
    pyModule1 = py.importlib.import_module('src.nn_inference');
    pyModule2 = py.importlib.import_module('src.nn_jacobian');
end

% Nonlinear force vector with NN needs input in cosine-sine form
NN_id = '2026-04-14_12-00-35';  % Identifier of the trained NN model
nn_input = [X(2:3).', X(6:7).'];
nn_output = double(pyModule1.evaluate_Duffing_nn_H3(NN_id, nn_input));
Fnl_cs = [0, nn_output(1:2), 0, 0, nn_output(3:4)];
Fnl_ce = [flipud(((Fnl_cs(2:2:end) + 1i * Fnl_cs(3:2:end))/2).'); ...
    Fnl_cs(1); ...
    ((Fnl_cs(2:2:end) - 1i * Fnl_cs(3:2:end))/2).'];

% Conversion of sine-cosine to complex-exponential representation
Q_ce = [flipud(X(2:2:end -1)+1i*X(3:2:end -1))/2; ...
    X(1); ...
    (X(2:2:end -1) -1i*X(3:2:end -1))/2];

% Excitation frequency
Om = X(end);

% P is the magnitude of the cosine forcing
Fex_ce = [zeros(H-1,1);P/2;0;P/2; zeros(H-1 ,1)];

% Dynamic force equilibrium
R_ce = ( -((-H:H)'*Om).^2 * mu + 1i*(-H:H)'*Om * zeta + kappa ).* Q_ce+...
    Fnl_ce-Fex_ce;

% Conversion of complex-exponential to sine-cosine representation
R = [real(R_ce(H+1)); ...
     reshape([2*real(R_ce(H+2:end)), -2*imag(R_ce(H+2:end))].', [], 1)];

nonlinear_term = 'NN';
evaluate_coefficients = 0;
dR = double(pyModule2.NN_jacobian_Duffing_H3(reshape(X(1:end-1), 1, []), mu, zeta, kappa, gamma, P, H, N, nonlinear_term, NN_id, evaluate_coefficients, Om));

% Conversion from sine-cosine to complex-exponential representation
n = 1;
I0 = 1:n; ID = n+(1:H);
IC = n+repmat(1:n,1,H)+n*kron(0:2:2*(H-1),ones(1,n)); IS = IC+n;
Q = zeros(n*(H+1),1);
Q(I0) = X(I0);
Q(ID) = X(IC)-1i*X(IS);

% Align to AFT convention (ordering + scaling)
perm_row = [1 2 4 6 3 5 7];
alpha = 0.5;
R = alpha * R(perm_row);
if nargout > 1
    dR = alpha * dR(perm_row, :);
end

end