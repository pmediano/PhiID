function [ redred, localred ] = DoubleRedundancyMMI(varargin)
%%DOUBLEREDUNDANCYMMI Compute the PhiID double-redundancy of input data,
% assuming it follows a Gaussian distribution and using the MMI PID.
%
% NOTE: assumes JIDT has been already added to the javaclasspath.
%
%   R = DOUBLEREDUNDANCYMMI(X, TAU), where X is a D-by-T data matrix of D
%   dimensions for T timesteps, and TAU is an integer integration timescale,
%   computes the double-redundancy across the minimum information bipartition
%   (MIB) of X. If TAU is not provided, it is set to 1.
%
%   R = DOUBLEREDUNDANCYMMI(X1, X2, Y1, Y2), where all inputs are matrices with
%   the same number of columns (i.e. same number of samples), computes the
%   double-redundancy of the mutual info between them, I(X1, X2; Y1, Y2).
%
%   [R, L] = DOUBLEREDUNDANCYMMI(...) returns the local double-redundancy
%   values for each sample in the input.
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
  error('Wrong number of arguments. See `help DoubleRedundancyMMI` for help.');
end

redred = mean(R(isfinite(R)));

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


% Stack data for easier handling (also scale to unit variance for numerical stability)
renorm = @(X) X./repmat(std(X')', [1, T]);
src = {renorm(X1), renorm(X2)};
tgt = {renorm(Y1), renorm(Y2)};


% Take double-redundancy as the minimum MI between either src or tgt
redred = 1e9*ones([T, 1]);  % Set to a large, but finite value
miCalc = javaObject('infodynamics.measures.continuous.gaussian.MutualInfoCalculatorMultiVariateGaussian');
for i=1:length(src)
  for j=1:length(tgt)
    s = src{i};
    t = tgt{j};

    miCalc.initialise(size(s, 1), size(t, 1));
    miCalc.setObservations(octaveToJavaDoubleMatrix(s'), octaveToJavaDoubleMatrix(t'));
    mi = miCalc.computeLocalOfPreviousObservations();

    if mean(mi(isfinite(mi))) < mean(redred(isfinite(mi)))
      redred = mi;
    end
  end
end

end

