function [ redred, localred ] = DoubleRedundancyMMIDiscrete(varargin)
%%DOUBLEREDUNDANCYMMIDISCRETE Compute the PhiID double-redundancy of discrete
% input data using the MMI PID. It uses a plug-in MI estimator first to find
% the MIB, and then uses a data-efficient quasi-Bayesian estimator to compute
% the double-redundancy value.
%
% NOTE: assumes JIDT has been already added to the javaclasspath.
%
%   R = DOUBLEREDUNDANCYMMIDISCRETE(X, TAU), where X is a D-by-T data matrix of
%   D dimensions for T timesteps, and TAU is an integer integration timescale,
%   computes the double-redundancy across the minimum information bipartition
%   (MIB) of X. If TAU is not provided, it is set to 1.
%
%   R = DOUBLEREDUNDANCYMMIDISCRETE(X1, X2, Y1, Y2), where all inputs are
%   matrices with the same number of columns (i.e. same number of samples),
%   computes the double-redundancy of the mutual info between them, I(X1, X2;
%   Y1, Y2).
%
%   [R, L] = DOUBLEREDUNDANCYMMIDISCRETE(...) returns the local
%   double-redundancy values for each sample in the input. (NOTE: not
%   available for the NSB estimator.)
%
% If input data is discrete-compatible (as per ISDISCRETE), it is passed
% directly to the underlying information-theoretic calculators. If it isn't
% (e.g. if it is real-valued data), it is mean-binarised first.
%
% Reference:
%   Mediano*, Rosas*, Carhart-Harris, Seth and Barrett (2019). Beyond
%   Integrated Information: A Taxonomy of Information Dynamics Phenomena.
%
% Pedro Mediano, Jan 2021

if nargin == 1
  R = private_TDMMI(varargin{1});
elseif nargin == 2
  R = private_TDMMI(varargin{1}, varargin{2});
elseif nargin == 4
  R = private_FourVectorMMI(varargin{1}, varargin{2}, varargin{3}, varargin{4});
else
  error('Wrong number of arguments. See `help DoubleRedundancyMMIDiscrete` for help.');
end

redred = mean(R);

if nargout > 1
  localred = R;
end

end


%*********************************************************
%*********************************************************
function [ redred ] = private_TDMMI(X, tau)

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

% Binarise the data, if not already discrete
if ~isdiscrete(X)
  X = 1*(X > mean(X, 2));
end


% Use JIDT to compute Phi and MIB
phiCalc = javaObject('infodynamics.measures.discrete.IntegratedInformationCalculatorDiscrete', 2, size(X, 1));
if tau > 1
  phiCalc.setProperty(phiCalc.PROP_TAU, num2str(tau));
end
phiCalc.setObservations(octaveToJavaIntMatrix(X'));
phi = phiCalc.computeAverageLocalOfObservations();
mib = phiCalc.getMinimumInformationPartition();

% Extract MIB partition indices
p1 = str2num(mib.get(0).toString()) + 1;
p2 = str2num(mib.get(1).toString()) + 1;

redred = private_FourVectorMMI(X(p1, 1:end-tau), X(p2, 1:end-tau), ...
                                X(p1, 1+tau:end), X(p2, 1+tau:end));

end


%*********************************************************
%*********************************************************
function [ redred ] = private_FourVectorMMI(X1, X2, Y1, Y2)

% Argument checks and parameter initialisation
T = size(X1, 2);
if size(X2, 2) ~= T || size(Y1, 2) ~= T || size(Y2, 2) ~= T
  error('All input vectors must have the same length');
end


% Binarise data (if not already discrete) and stack for easier handling
binarify = @(v) isdiscrete(v)*v + (~isdiscrete(v))*(v > mean(v, 2));
src = {binarify(X1), binarify(X2)};
tgt = {binarify(Y1), binarify(Y2)};


% Take double-redundancy as the minimum MI between either src or tgt
redred = inf([T, 1]);
for i=1:length(src)
  for j=1:length(tgt)

    x = ensure_combined(src{i});
    y = ensure_combined(tgt{j});

    miCalc = javaObject('infodynamics.measures.discrete.MutualInformationCalculatorDiscrete', max(x)+1, max(y)+1, 0);
    miCalc.initialise();
    miCalc.addObservations(x', y');
    mi = miCalc.computeLocalFromPreviousObservations(x', y');

    if mean(mi) < mean(redred)
      redred = mi;
    end
  end
end

end


%*********************************************************
%*********************************************************
function [ V ] = ensure_combined(U)
  if size(U, 1) == 1
    V = U;
  else
    [~,~,V] = unique(U', 'rows');
    V = V(:)' - 1;
  end
end


