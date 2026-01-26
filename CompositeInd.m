%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                   Model-based Composite Indicator                     %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [gCI,cCI] = CompositeInd(Y,NNLFLG,varargin)


% 'MeasureMod' ->   Default 'Reflective'. It returns a reflective estimation
%                   model: starting point = 1 PC.
%                   If 'Formative' returns a formative estimation
%                   model.
% 'DataMatrix' ->   If 'MeasureMod' is 'Formative', the data matrix is not
%                   needed. Otherwise, it is compulsory. 
% 'LoadMatrix' ->   If 'MeasureMod' is 'Formative', the loading matrix is not
%                   needed. Otherwise, it is compulsory.
% 'ObjFunc'    ->   If 'MeasureMod' is 'Formative', the last value of the 
%                   objective function is not needed. Otherwise, it is compulsory.
%                
% 'MaxIter'    ->   an integer value indicating the maximum number of
%                   iterations of the algorithm.
% 'ConvToll'   ->   an arbitrary small values indicating the convergence
%                   tollerance of the algorithm, Default '1e-9'.


pnames = {'MeasureMod' 'DataMatrix' 'MaxIter'};
dflts =  {'Reflective'      []         100};
[MeasureMod,DataMatrix,MaxIter] = internal.stats.parseArgs(pnames, dflts, varargin{:});

% MeasureMod %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if ~isempty(MeasureMod)
    if ischar(MeasureMod)
       MeasureNames = {'Formative', 'Reflective'};
       MMs = strcmpi(MeasureMod,MeasureNames);
           if sum(MMs) == 0
              error(['Invalid value for the ''MeasureMod'' parameter: '...
                     'choices are ''Formative'' or ''Reflective''.']);
           end
       MeasureMod = MeasureNames{MMs}; 
    else  
        error(['Invalid value for the ''MeasureMod'' parameter: '...
               'choices are ''Formative'' or ''Reflective''.']);
    end
else 
    error(['Invalid value for the ''MeasureMod'' parameter: '...
           'choices are ''Formative'' or ''Reflective''.']);
end

MMFLG = 0;
MMs = strcmpi(MeasureMod,'Formative');
if sum(MMs) == 1
       MMFLG = 1;
end
% end MeasureMod %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% DataMatrix %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if MMFLG == 1 && isempty(DataMatrix)
    error(['Invalid value for the ''DataMatrix'' parameter: '...
        'if the model is ''Formative'', a data matrix MUST BE defined in the function inputs.']);
elseif ~isnumeric(DataMatrix)
    error(['Invalid value for the ''DataMatrix'' parameter: '...
        'it MUST be a numeric matrix.']);
elseif MMFLG == 1 && ~isempty(DataMatrix) && isnumeric(DataMatrix)
    X = DataMatrix;
end
% end DataMatrix %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% MaxIter %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if ~isempty(MaxIter)  
    if isnumeric(MaxIter)
       if (MaxIter < 0) || (MaxIter > 1000) 
       error('MaxIter must be a value in the interval [0,1000]');
       end
    else
       error('Invalid Number of Max Iterations');
    end
else
    error('Invalid Number of Max Iterations')
end
% end MaxIter %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if MMFLG == 0 
    [cCI,gCI,~] = ACP(Y,NNLFLG);
else
    % Inizialization
    [~,g,] = ACP(X,NNLFLG);
    % Regression
    cCI = pinv(Y)*g;
    gCI = Y*cCI;
end


% ------------------------ Local Functions---------------------------------

function [a,y,itr] = ACP(Xr,NNLFLG)
% ACP function with the power method.

maxit=300;
[n,Q]=size(Xr);
a = rand(Q,1); 
tol = 1e-9;
error = inf;
last = inf;
itr = 0;
while ~(abs(last-error)<error*tol) && itr<=maxit
    itr = itr+1;
    y = Xr*a./(a'*a);
    a=Xr'*y./(y'*y);
    if NNLFLG == 1   % if loadings must be non negative
        psa=find(a>=0);                             %
        if size(psa) < Q                            %
            y=Xr(:,psa)*a(psa)./(a(psa)'*a(psa));   %
            as=find(a<0);                           % if non negative
            a(psa)=Xr(:,psa)'*y./(y'*y);            %
            a(as)=0;                                %
        end                                         %
    end
    last = error;    
    e = y-Xr*a;
    error = e'*e/n;
end
a=a./sqrt(a'*a);
y=Xr*a;



