function [ A ] = PhiIDFull(varargin)
%%PHIIDFULL Computes full PhiID decomposition of input data, assuming it
% follows a multivariate Gaussian distribution and using the MMI PhiID.
%
%   A = PHIIDFULL(X, TAU), where X is a D-by-T data matrix of D dimensions for
%   T timesteps, and TAU is an integer integration timescale, computes the
%   PhiID decomposition of the time-delayed mutual information of X. If TAU is
%   not provided, it is set to 1. If D > 2, PhiID is calculated across the
%   minimum information bipartition (MIB) of the system.
%
%   A = PHIIDFULL(X1, X2, Y1, Y2), where all inputs matrices with T columns,
%   computes the PhiID decomposition of the mutual information between them,
%   I(X1, X2; Y1, Y2).
%
% In all cases, results are returned in a struct A with all integrated
% information atoms. Atoms are named with a three-char string of the form QtP,
% where Q and P are one of r, x, y, or s (redundancy, unique X, unique Y or
% synergy, respectively). For example:
%
%            A.rtr is atom {1}{2}->{1}{2}
%            A.xty is atom {1}->{2}
%            A.stx is atom {12}->{1}
%            ...
%
% Reference:
%   Mediano*, Rosas*, Carhart-Harris, Seth and Barrett (2019). Beyond
%   Integrated Information: A Taxonomy of Information Dynamics Phenomena.
%
% Pedro Mediano and Andrea Luppi, Jan 2021

% Find JIDT and add relevant paths
if ~any(~cellfun('isempty', strfind(javaclasspath('-all'), 'infodynamics')))
  p = strrep(mfilename('fullpath'), 'PhiIDFull', '');
  if exist([p, 'private/infodynamics.jar'], 'file')
    javaaddpath([p, 'private/infodynamics.jar']);
  else
    error('Unable to find JIDT (infodynamics.jar).');
  end
end

if nargin == 1
  atoms = private_TDPhiID(varargin{1});
elseif nargin == 2
  atoms = private_TDPhiID(varargin{1}, varargin{2});
elseif nargin == 4
  atoms = private_FourVectorPhiID(varargin{1}, varargin{2}, varargin{3}, varargin{4});
else
  error('Wrong number of arguments. See `help PhiIDFull` for help.');
end

if any(structfun(@(x) any(~isfinite(x)), atoms))
  warning('PhiID:Outlier', 'Outliers detected in PhiID computation. Results may be biased.');
end

A = structfun(@(x) mean(x(isfinite(x))), atoms, 'UniformOutput', 0);

end


%*********************************************************
%*********************************************************
function [ atoms ] = private_TDPhiID(X, tau)

% Argument checks and parameter initialisation
if isempty(X) || ~ismatrix(X)
  error('Input must be a 2D data matrix');
end
[D, T] = size(X);
if T <= D
  error(sprintf(['Your matrix has %i dimensions and %i timesteps. ', ...
        'If this is true, you cant compute a reasonable covariance matrix. ', ...
        'If it is not true, you may have forgotten to transpose the matrix'], D, T));
end
if nargin < 2 || isempty(tau)
  tau = 1;
end
integer_tau = ~isinf(tau) & floor(tau) == tau;
if ~integer_tau || tau < 1
  error('Timescale tau needs to be a positive integer.');
end

% Create copy of the data scaled to unit variance (for numerical stability)
sX = X./repmat(std(X')', [1, T]);


% Use JIDT to compute Phi and MIB
phiCalc = javaObject('infodynamics.measures.continuous.gaussian.IntegratedInformationCalculatorGaussian');
if tau > 1
  phiCalc.setProperty(phiCalc.PROP_TAU, num2str(tau));
end
phi = phiCalc.compute(octaveToJavaDoubleMatrix(sX'));
mib = phiCalc.getMinimumInformationPartition();

% Extract MIB partition indices
p1 = str2num(mib.get(0).toString()) + 1;
p2 = str2num(mib.get(1).toString()) + 1;

% Stack data and call full PhiID function
atoms = private_FourVectorPhiID(sX(p1,1:end-tau), sX(p2,1:end-tau), ...
                                sX(p1,1+tau:end), sX(p2,1+tau:end));

end


%*********************************************************
%*********************************************************
function [ atoms ] = private_FourVectorPhiID(X1, X2, Y1, Y2)

% Argument checks and parameter initialisation
checkmat = @(v) ~isempty(v) && ismatrix(v);
if ~(checkmat(X1) && checkmat(X2) && checkmat(Y1) && checkmat(Y2))
  error('All inputs must be non-empty data matrices');
end
T = size(X1, 2);
if size(X2, 2) ~= T || size(Y1, 2) ~= T || size(Y2, 2) ~= T
  error('All input matrices must have the same number of columns');
end

RedFun = @RedundancyMMI;
DoubleRedFun = @DoubleRedundancyMMI;

% Indices of source and target variables in the joint covariance matrix
p1 = 1:size(X1, 1);
p2 = p1(end)+1:p1(end)+size(X2, 1);
t1 = p2(end)+1:p2(end)+size(Y1, 1);
t2 = t1(end)+1:t1(end)+size(Y2, 1);
D = t2(end);


% Create copy of the data scaled to unit variance (for numerical stability)
X = [X1; X2; Y1; Y2];
sX = X./repmat(std(X')', [1, T]);

% Compute mean and covariance for all the data
% (to be used by the local IT functions below)
S = cov(sX');
mu = mean(sX');
assert(all(size(mu) == [1, D]) && all(size(S) == [D, D]));

% Define local information-theoretic functions
h = @(idx) -log(mvnpdf(sX(idx,:)', mu(idx), S(idx,idx)));
mi  = @(src, tgt) h(src) + h(tgt) - h([src, tgt]);

% Pre-compute entropies necessary for all IT quantities
h_p1 = h(p1);
h_p2 = h(p2);
h_t1 = h(t1);
h_t2 = h(t2);

h_p1p2 = h([p1 p2]);
h_t1t2 = h([t1 t2]);
h_p1t1 = h([p1 t1]);
h_p1t2 = h([p1 t2]);
h_p2t1 = h([p2 t1]);
h_p2t2 = h([p2 t2]);

h_p1p2t1 = h([p1 p2 t1]);
h_p1p2t2 = h([p1 p2 t2]);
h_p1t1t2 = h([p1 t1 t2]);
h_p2t1t2 = h([p2 t1 t2]);

h_p1p2t1t2 = h([p1 p2 t1 t2]);

% Compute PhiID quantities as entropy combinations
Ixytab = h_p1p2 + h_t1t2 - h_p1p2t1t2;

Ixta = h_p1 + h_t1 - h_p1t1;
Ixtb = h_p1 + h_t2 - h_p1t2;
Iyta = h_p2 + h_t1 - h_p2t1;
Iytb = h_p2 + h_t2 - h_p2t2;

Ixyta = h_p1p2 + h_t1 - h_p1p2t1;
Ixytb = h_p1p2 + h_t2 - h_p1p2t2;
Ixtab = h_p1 + h_t1t2 - h_p1t1t2;
Iytab = h_p2 + h_t1t2 - h_p2t1t2;

Rxyta  = RedFun(sX, p1, p2, t1, Ixta, Iyta, Ixyta);
Rxytb  = RedFun(sX, p1, p2, t2, Ixtb, Iytb, Ixytb);
Rxytab = RedFun(sX, p1, p2, [t1 t2], Ixtab, Iytab, Ixytab);
Rabtx  = RedFun(sX, t1, t2, p1, Ixta, Ixtb, Ixtab);
Rabty  = RedFun(sX, t1, t2, p2, Iyta, Iytb, Iytab);
Rabtxy = RedFun(sX, t1, t2, [p1 p2], Ixyta, Ixytb, Ixytab);

% Compute double-redundancy with corresponding function
[~, rtr] = DoubleRedFun(sX(p1,:), sX(p2,:), sX(t1,:), sX(t2,:));

% Assemble and solve system of equations
reds = [rtr Rxyta Rxytb Rxytab Rabtx Rabty Rabtxy ...
        Ixta Ixtb Iyta Iytb Ixyta Ixytb Ixtab Iytab Ixytab];

M = [1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0; % rtr
     1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0; % Rxyta
     1 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0; % Rxytb
     1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0; % Rxytab
     1 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0; % Rabtx
     1 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0; % Rabty
     1 0 0 0 1 0 0 0 1 0 0 0 1 0 0 0; % Rabtxy
     1 1 0 0 1 1 0 0 0 0 0 0 0 0 0 0; % Ixta
     1 0 1 0 1 0 1 0 0 0 0 0 0 0 0 0; % Ixtb
     1 1 0 0 0 0 0 0 1 1 0 0 0 0 0 0; % Iyta
     1 0 1 0 0 0 0 0 1 0 1 0 0 0 0 0; % Iytb
     1 1 0 0 1 1 0 0 1 1 0 0 1 1 0 0; % Ixyta
     1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0; % Ixytb
     1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0; % Ixtab
     1 1 1 1 0 0 0 0 1 1 1 1 0 0 0 0; % Iytab
     1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1]; % Ixytab

partials = linsolve(M, reds');


% Sort the results and return
atoms = [];
atoms.rtr = partials(1,:);
atoms.rtx = partials(2,:);
atoms.rty = partials(3,:);
atoms.rts = partials(4,:);
atoms.xtr = partials(5,:);
atoms.xtx = partials(6,:);
atoms.xty = partials(7,:);
atoms.xts = partials(8,:);
atoms.ytr = partials(9,:);
atoms.ytx = partials(10,:);
atoms.yty = partials(11,:);
atoms.yts = partials(12,:);
atoms.str = partials(13,:);
atoms.stx = partials(14,:);
atoms.sty = partials(15,:);
atoms.sts = partials(16,:);

end


%*********************************************************
% Utility functions to compute basic information-theoretic measures
%*********************************************************
function [ res ] = logdet(A)
  res = 2*sum(log(diag(chol(A))));
end
function [ res ] = h(S, idx)
  res = 0.5*length(idx)*log(2*pi*exp(1)) + 0.5*logdet(S(idx,idx));
end
function [ res ] = mi(S, src, tgt)
  res = h(S, src) + h(S, tgt) - h(S, [src, tgt]);
end


%*********************************************************
% A few PID (single-target) redundancy functions
%*********************************************************
function [ R ] = RedundancyMMI(bX, src1, src2, tgt, mi1, mi2, mi12)
  if mean(mi1) < mean(mi2)
    R = mi1;
  else
    R = mi2;
  end
end

