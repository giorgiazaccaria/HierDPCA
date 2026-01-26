%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Disjoint Principal Component Analysis %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Maurizio Vichi
% October 2012, revised November 2013, revised for non ngative November 2014 
% revised for improving parameters September 2015

% problem maximize ||XBV||^2
% subject to
% A=BV where
% V'B'BV=I
% and V is binary and row stochastic
% B = diagonal matrix

function [Vdpca,Adpca,Ydpca,fdpca,indpca] = DPCA(X,K,varargin)
% X (n X J) data matrix
% V (J x K) membership matrix for clustering variables
% n = number of objects
% J = number of variables
% K = number of factor clusters
%
% eventually to display a correlation matrix
% imagesc(Sx)
% colormap(flipud(hot));
% colorbar

% INPU�T
% X (n X J) data matrix
% K = number of factors
%

%
% Optional parameters:
%
% 'Stats'    ->    Default value: 'on', print the statistics of the fit of the
%                  model.
%                  If 'off' Statistics are not printed (used for simulation
%                  studies)
% 'Stand'    ->    Default value 'on', standardize variables and therefore compute DPCA 
%                  on the correlation matrix.
%                  If 'off' does not standardize variables and therefore
%                  compute DFA on the variance-covariance matrix
% 
% 'Rndst'    ->    an integer values indicating the intital random starts.
%                  Default '20' thus, repeat the anaysis 20 times and retain the
%                  best solution.
% 'MaxIter'  ->    an integer value indicating the maximum number of
%                  iterations of the algorithm
%                  Default '100'.
% 'ConvToll' ->    an arbitrary small values indicating the convergence
%                  tollerance of the algorithm, Default '1e-9'.
% 'Constr'   ->    Default Constr=[] i.e., no constraints.
%                  Vector Constr (J x 1) indicates for each variable if the
%                  variable is constrained to be in a fixed class. 
%                  Constr(j)= number of class class (c(j) between 1 and K) 
%                  Constr(j)=0 no constraints for variable j.      
% 'NN'       ->    Defaulf 'off', loadings are free
%                  If 'on' loadings are constrained to be non negative

% OUTPUT
% Vdfa (J x K)   binary membership matrix indicating the classes of variables
%                Vdfa(j,h)=1 if variable j belongs to class h,  
%                Vdfa(j,h)=0 otherwise.
% Adfa  (J x K)  loading matrix 
% Psifa (J x 1)  vector of the errors
% Ydfa  (n x K)  Factor scores 
% fdfa   (1)     discepancy fantion at convergence   
% indfa  (1)     number of time optimal solution was found
%
% n = numenr of objects
% J = number of variables
%
% OPTIONS
%
% Set optional parameters
%
% Required parameters: X and K

%
%
% Set optional parameters
%
% Required parameters: X and K

% initialization
rng(1);
[n,J]=size(X);
VC=eye(K);
opts.disp=0;

% centrering matrix
Jc=eye(n)-(1./n)*ones(n);

if nargin < 2
   error('Too few inputs');
end

if ~isempty(X)
    if ~isnumeric(X)
        error('Invalid data matrix');
    end  
    if min(size(X)) == 1
    error(message('Disjoint Factor Analysis:NotEnoughData'));
end

else
    error('Empty input data matrix');
end

if ~isempty(K)
    if isnumeric(K)
        if K > J 
              error('The number of latent factors larger that the number of variables');
           end    
    elseif K < 1 
              error('Invalid number of latent factors');
    end
else
    error('Empty input number of latent factors');
end



% Optional parameters   
pnames = {'Stats' 'Stand' 'Rndst' 'MaxIter' 'ConvToll' 'Constr' 'CrsLoad' 'NNL'};
dflts =  { 'on'    'on'     20       100       1e-9    zeros(J,1) 'off' 'off'};
[Stats,Stand,Rndst,MaxIter,ConvToll,Constr,CrsLoad,NNL] = internal.stats.parseArgs(pnames, dflts, varargin{:});


%if ~isempty(eid)
%    error(sprintf('Disjoint Factor Analysis; %s',eid), emsg);
%end


% Statistics %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if ~isempty(Stats)
    if ischar(Stats)
       StatsNames = {'off', 'on'};
       js = strcmpi(Stats,StatsNames);
           if sum(js) == 0
              error(['Invalid value for the ''Statistics'' parameter: '...
                     'choices are ''on'' or ''off''.']);
           end
       Stats = StatsNames{js}; 
    else  
        error(['Invalid value for the ''Statistics'' parameter: '...
               'choices are ''on'' or ''off''.']);
    end
else 
    error(['Invalid value for the ''Statistics'' parameter: '...
           'choices are ''on'' or ''off''.']);
end
% end statistics %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Standardization %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if ~isempty(Stand)
    if ischar(Stand)
       StandNames = {'off', 'on'};
       js = strcmpi(Stand,StandNames);
           if sum(js) == 0
              error(['Invalid value for the ''Standardization'' parameter: '...
                     'choices are ''on'' or ''off''.']);
           end
       Stand = StandNames{js}; 
       switch Stand
           
       case 'off'
           Xs = X;            
       case 'on'
           Xs = zscore(X,1);
       end
    else  
        error(['Invalid value for the ''standardization'' parameter: '...
               'choices are ''on'' or ''off''.']);
    end
else 
    error(['Invalid value for the ''standardization'' parameter: '...
           'choices are ''on'' or ''off''.']);
end
% end Standardization %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Rndst %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if ~isempty(Rndst)  
    if isnumeric(Rndst)
       if (Rndst < 0) || (Rndst > 1000) 
       error('Rndst must be a value in the interval [0,1000]');
       end
    else
       error('Invalid Number of Random Starts');
    end
else
    error('Invalid Number of Random Starts')
end
% end Rndst %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% MAxIter %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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


% ConvToll %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if ~isempty(ConvToll)  
    if isnumeric(ConvToll)
       if (ConvToll < 0) || (ConvToll > 0.1) 
       error('ConvToll must be a value in the interval [0,0.1]');
       end
    else
       error('Invalid Convergence Tollerance');
    end
else
    error('Invalid Convergence Tollerance')
end
% end ConvToll %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Constr %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if ~isempty(Constr)  
    if isnumeric(Constr)
        for j=1:J
            if (Constr(j) < 0) || (Constr(j) > K) 
                error('Constr must be a value in the interval [0,K]');
            end
        end  
    else
       error('Invalid Constraint');
    end
else
    error('Invalid Constraint')
end
% end Constr %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Cross-Loadings %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if ~isempty(CrsLoad)
    if ischar(CrsLoad)
       CrsLoadNames = {'off', 'on'};
       jcl = strcmpi(CrsLoad,CrsLoadNames);
           if sum(jcl) == 0
              error(['Invalid value for the ''Cross-loadings'' parameter: '...
                     'choices are ''on'' or ''off''.']);
           end
       CrsLoad = CrsLoadNames{jcl}; 
    else  
        error(['Invalid value for the ''CrsLoad'' parameter: '...
               'choices are ''on'' or ''off''.']);
    end
else 
    error(['Invalid value for the ''CrsLoad'' parameter: '...
           'choices are ''on'' or ''off''.']);
end
CrsLoadFLG=0;
js=strcmpi(CrsLoad,'on');
if sum(js)==1
       CrsLoadFLG=1;
end
% end CrsLoad %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% NonNegative-Loadings %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if ~isempty(NNL)
    if ischar(NNL)
       NNLNames = {'off', 'on'};
       jcl = strcmpi(NNL,NNLNames);
           if sum(jcl) == 0
              error(['Invalid value for the ''NonNegative-loadings'' parameter: '...
                     'choices are ''on'' or ''off''.']);
           end
       NNL = NNLNames{jcl}; 
    else  
        error(['Invalid value for the ''NNL'' parameter: '...
               'choices are ''on'' or ''off''.']);
    end
else 
    error(['Invalid value for the ''CrsLoad'' parameter: '...
           'choices are ''on'' or ''off''.']);
end

NNLFLG=0;
js=strcmpi(NNL,'on');
if sum(js)==1
       NNLFLG=1;
end

% end NNL %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% compute var-covar matrix %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

S=cov(Xs,1);

% compute Total Variance   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
st=(1./n)*sum(sum(Xs.^2));

JJ=[1:J]';    
zJ=zeros(J,1);

% Start the algorithm      %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for loop=1:Rndst
    it=0;
    flg=1;
    while flg > 0
        V=randPU(J,K);     % initial V and A
        % if there are Constraints on the variables the initial V has to
        % satisfy constraints
        for j=1:J
            if Constr(j)>0
                V(j,:)=VC(Constr(j),:);
            end    
        end
        flg=sum(find(sum(V)==0));
    end

    A=zeros(J,K);

    for g=1:K
            ibCg=V(:,g); 
            JCg=[JJ(ibCg==1)];
            %Sg=S(JCg,JCg);

            if sum(ibCg)>1
                %[a,c]=eigs(Sg,1,'lm',opts);
                [a,y,itr] = ACP(Xs(:,JCg),NNLFLG);
                %a=a./sqrt(a'*a);
                A(JCg,g)=a;
            else
                A(JCg,g)=1;
            end
    end
    A0=A;
    % 
    % initial scores
    %
    Y=Xs*A;
    %
    f0=trace((1./n)*(Y'*Y));
    fmax=0;

    % iteration phase
    fdif=2*eps;
    for it=1:MaxIter
          
        % update V and A
        for j=1:J
            posmax=JJ(V(j,:)==1);
            if Constr(j) == 0
                for g=1:K
                    V(j,:)=VC(g,:);
                    if sum(V(:,posmax))>0
                        ibCg=V(:,g);           % classe attuale V
                        ibCpm=V(:,posmax);     % classe vecchia V
                        JCg=[JJ(ibCg==1)];
                        JCpm=[JJ(ibCpm==1)];
                        if sum(ibCg)>1
                            [a,y,itr] = ACP(Xs(:,JCg),NNLFLG);
                            if sum(a)<0
                                a=-a;
                            end
                            A(:,g)=zJ;
                            A(JCg,g)=a;
                        else
                            A(:,g)=zJ;
                            A(JCg,g)=1;
                        end
                        if sum(ibCpm)>1
                            [aa,y,itr] = ACP(Xs(:,JCpm),NNLFLG);
                            if sum(aa)<0
                                aa=-aa;
                            end
                            A(:,posmax)=zJ;  
                            A(JCpm,posmax)=aa;
                        else
                            A(:,posmax)=zJ;  
                            A(JCpm,posmax)=1;
                        end            
                        Y=Xs*A;
                        f=trace((1./n)*(Y'*Y));
                        if f > fmax
                            fmax=f;
                            posmax=g;
                            A0=A;
                        else
                            A=A0;
                        end
                    end
                end
            end
            V(j,:)=VC(posmax,:);
        end
        
        Y=Xs*A; 
        f=trace((1./n)*(Y'*Y));
        fdif = f-f0;
        
        if fdif > ConvToll 
            f0=f; fmax=f0;A0=A; 
        else
            break
        end
    end
  disp(sprintf('DPCA: Loop=%g, Explained variance=%g, iter=%g, fdif=%g',loop,f./st*100, it,fdif))   
       if loop==1
            Vdpca=V;
            Adpca=A;
            Ydpca=Xs*Adpca;
            fdpca=f;
            loopdpca=1;
            indpca=it;
            fdifo=fdif;
        end
   if f > fdpca
       Vdpca=V;
       fdpca=f;
       Adpca=A;
       Ydpca=Xs*Adpca;
       loopdpca=loop;
       indpca=it;
       fdifo=fdif;
   elseif abs(f -fdpca)<0.0001 & loop>1
%       break
   end

end
% sort the final solution in descend order of variance
varYdpca=var(Ydpca,1);
% [~,ic]=sort(varYdpca, 'descend');
% Adpca=Adpca(:,ic);
% Vdpca=Vdpca(:,ic);
% Ydpca=Ydpca(:,ic); 

% compute the the variance of the second component
% and variance of the error 
s2=0;
e2k=zeros(K,1);
cro=zeros(K,1);
for g=1:K
    ibCg=Vdpca(:,g); 
    if sum(ibCg)>1
        JCg=[JJ(ibCg==1)];
        cro(g)=CronbachAlpha(Xs(:,JCg));
        Sg=S(JCg,JCg);
        [U,L]=eig(Sg);
%         [l,il]=sort(diag(L), 'descend');
%         L=diag(l);
%         U=U(:,il);
        L = diag(sort(diag(L),'descend'));
        e2k(g)=L(2,2);
    else
        e2k(g)=0;
        cro(g)=1;
    end
    %s2 = s2+sum(l(2:size(Sg,1)));
    %s2 = s2+sum(L(2:size(Sg,1)));
end
%s2=s2./(J-K);

% Statistics %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

js=strcmpi(Stats,'off');
if sum(js) == 0

    disp(sprintf('\n \n Disjoint PCA (Final): Explained variance=%g, loopdpca=%g, iter=%g, fdif=%g',fdpca./st*100, loopdpca, indpca,fdifo))
    disp(sprintf('\n \n Disjoint Principal Component Analysis results'))
    disp(sprintf('\n \nFactor            Variance Explained     Perc. Expl. Var.   Cumulated Var.  Perc. Expl. Var.'))
    VarCum=0;
    v = zeros(K,1);
    for k=1:K
        v(k)= trace(1/n*Ydpca(:,k)'*Ydpca(:,k));
        VarCum = VarCum + v(k);
        disp(sprintf('factor   (%g)      %f               %f          %f        %f  ', k, v(k), v(k)./st*100, VarCum, VarCum./st*100))
    end
    disp(sprintf('\n \n Loading Matrix (Manifest Variables x Latent Variables)'))
    Adpca
    for k=1:K
        disp(sprintf('Unidimensionality Assessment: variance of the second component of class(%g)    %f', k,e2k(k)))
   end
    for k=1:K
        disp(sprintf('Reliability Assessment: Alpha of Cronbach of class(%g)    %f', k,cro(k)))
    end
    disp(sprintf('\n \n         Path           Corr coef       Std Error        Pr(p>|Z|)        Var Error        Communality' ))
    for j=1:J
        p=zeros(J,1);
        %p=zeros(J,2);
        for k=1:K
            %p(j,:)= normcdf([-1*abs(AA(j,1)/SD(j,j).^0.5) abs(AA(j,1)/SD(j,j).^0.5)]);
            if Adpca(j,k)~=0
                p(j)= r2pv(corr(Xs(:,j),Ydpca(:,k)),n);
                disp(sprintf('X(%g) <-- factor(%g)     % f        %f         %f         %f         %f', j, k, corr(Xs(:,j),Ydpca(:,k)), s2.^0.5./sqrt(n), p(j), s2, corr(Xs(:,j),Ydpca(:,k)).^2))    
            end
        end
    end
end



% --------------End of Main Function---------------------------------------


% ------------- Local Subfunctions-----------------------------------------
function [a,y,itr] = ACP(Xr,NNLFLG);
maxit=300;
[n,Q]=size(Xr);
a = rand(Q,1); 
tol = 1e-9;
error = inf;
last = inf;
itr = 0;
while ~(abs(last-error)<error*tol) & itr<=maxit
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


 function p=r2pv(r,n)
%
% 	p=r2pv(r,n)
%
% r = estimated correlation coefficient (IE |r| <= 1)
%   = (1/n)*(x'*y) for col vectors x,y of length n
% n = no. samples used
% p = P-value based on |r| (two sided) with rho=0 (null case)
%
% NOTES: following Cramer, p.400, convert r to a t and use what we have for t 
if n < 3
    error('n < 3');
end
if r==1. 
    p=0; 
    return;
end
t=sqrt(n-2)*r/(sqrt(1-r*r)); 	% this is t with n-2 d.f.
t=abs(t);							% use |t| for two sided P-value
p=2*(1-tcdf(t,n-2));

function [U]=randPU(n,c)

% generates a random partition of n objects in c classes
%
% n = number of objects
% c = number of classes
%
U=zeros(n,c);
U(1:c,:)=eye(c);

U(c+1:n,1)=1;
for i=c+1:n
    U(i,[1:c])=U(i,randperm(c));
end
U(:,:)=U(randperm(n),:);

function mri=mrand(N)
%
% modified rand index (Hubert & Arabie 1985, JCGS p.198)
%
n=sum(sum(N));
sumi=.5*(sum(sum(N').^2)-n);
sumj=.5*(sum(sum(N).^2)-n);
pb=sumi*sumj/(n*(n-1)/2);
mri=(.5*(sum(sum(N.^2))-n)-pb)/((sumi+sumj)/2-pb);

function [as,varargout] = CronbachAlpha(x)
% CronbachAlpha
% 
% Description:	calculate Cronbach's alpha for a set of psychometric measurements
% 
% Syntax:	[as,au] = CronbachAlpha(x)
% 
% In:
% 	x	- an nRep x nItem array of ratings, so that each row is the set of
%		  obvservations from one repetition and each column is the set of all
%		  observations for a given item
% 
% Out:
% 	as	- the standardized Cronbach's alpha
%	au	- the unstandardized Cronbach's alpha
% 
% Updated: 2012-09-24
% Copyright 2012 Alex Schlegel (schlegel@gmail.com).  This work is licensed
% under a Creative Commons Attribution-NonCommercial-ShareAlike 3.0 Unported
% License.
nItem	= size(x,2);

%logical array for selecting upper triangular part of the correlation and
%covariance matrices, where the good stuff is
	b	= triu(true(nItem),1);

%standardized alpha
	%pairwise correlations between items
		r	= corrcoef(x);
	%mean of the meaningful, non-redundant correlations
		r	= mean(r(b));
	
	as	= nItem*r/(1 + (nItem-1)*r);

%unstandardized alpha
if nargout>1
	%variance/covariance matrix
		vc	= cov(x);
	%mean variance (variances are along the diagonal)
		v	= mean(diag(vc));
	%mean covariance, not including variances
		c	= mean(vc(b));
	
	varargout{1}	= nItem*c/(v + (nItem-1)*c);
end

