function [hierV,of,Y_merge,A_merge,V_merge,CI,w,M,R2SCI,R2SCIh,cro,expvar] = HDPCA(X,Q,mlst,varargin)

% Hierarchical Disjoint Principal Component Analysis (HDPCA)

% Cavicchia, Vichi, Zaccaria
% March 2019

% INPUT
% Required parameters
% X (nxJ) data matrix
% Q number of factors 
% mlst random start
%
% Optional parameters
%
% 'LastACP'       ->   Default value: 'false', the reflective model for q = 1 is
%                      estimated minimizing ||Y_2-g'V'_1B_1||^2.
%                      If 'true', the reflective model for q = 1 is estimated minimizing 
%                      ||X-Y_1V'_1B_1||^2.
% 'NNL'           ->   Defaulf value: 'true', loadings are constrained to be non negative.
%                      If 'false' loadings are free.
% 'Stats'         ->   Default value: 'on', print the statistics of the fit of
%                      the model (for q = M,...,Q and M = 2,...,Q).
%                      If 'off' Statistics are not printed (used for simulation
%                      studies).
% 'normalization' ->   Character array. Normalization method to apply on X.
%                      Value: 
%                      - 'off'       --> No normalization, only centering.
%                      - 'standard'  --> Standardization: (X-mean)/SD
%                      - 'minmax'    --> (X-minX)/(maxX-minX).
%                      The default normalization is 'standard'.
% 'constrV'       ->   Vector of size J with values in [1,Q]. It binds the
%                      variables to the principal components.
% 'alpha'         ->   Significance level for the correlation test. 
%                      Default value = 0.05. 

% Inizialization %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
warning('off');
[n,J] = size(X);
vecJ = [1:J]';
vecQ = [1:Q]';
PPV = vecQ;
AV  = zeros(Q-1,1);
BV  = zeros(Q-1,1);
lbV = zeros(Q,1);
of{Q} = 0;
R2SCI = zeros(Q,1);
ica = 0;
jca = 0;
stopformative = 0;
test = 0;

% Input Errors %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if nargin < 3
   error('Too few inputs');
end

if ~isempty(X)
    if ~isnumeric(X)
        error('Invalid data matrix');
    end  
    if min(size(X)) == 1
    error(message('Hierarchical Factor Clustering:NotEnoughData'));
    end
else
    error('Empty input data matrix');
end

if ~isempty(Q)
    if isnumeric(Q)
        if Q > J 
              error('The number of latent factors larger than the number of variables');
        end
    elseif Q < 1 
              error('Invalid number of latent factors');
    end
else
    error('Empty input number of latent factors');
end
% End Input Errors %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Optional parameters
pnamesin = {'LastACP' 'NNL' 'Stats' 'normalization' 'constrV' 'alpha'};
dfltsin = {'false' 'true' 'off' 'standard' zeros(J,1) 0.05};
[LastACP,NNL,Stats,normalization,constrV,alpha] = internal.stats.parseArgs(pnamesin, dfltsin, varargin{:});

% LastACP %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if ~isempty(LastACP)
    if ischar(LastACP)
       ACPNames = {'true','false'};
       jacp = strcmpi(LastACP,ACPNames);
           if sum(jacp) == 0
              error(['Invalid value for the ''LastACP'' parameter: '...
                     'choices are ''true'' or ''false''.']);
           end
       LastACP = ACPNames{jacp}; 
    else  
        error(['Invalid value for the ''LastACP'' parameter: '...
               'choices are ''true'' or ''false''.']);
    end
end

ACPFLG = 0;
jsacp = strcmpi(LastACP,'true');
if sum(jsacp) == 1
       ACPFLG = 1;
end

% End LastACP %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% NonNegative-Loadings %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if ~isempty(NNL)
    if ischar(NNL)
       NNLNames = {'true','false'};
       jnnl = strcmpi(NNL,NNLNames);
           if sum(jnnl) == 0
              error(['Invalid value for the ''NonNegative-loadings'' parameter: '...
                     'choices are ''true'' or ''false''.']);
           end
       NNL = NNLNames{jnnl}; 
    else  
        error(['Invalid value for the ''NonNegative-loadings'' parameter: '...
               'choices are ''true'' or ''false''.']);
    end
end

NNLFLG = 0;
js = strcmpi(NNL,'true');
if sum(js) == 1
       NNLFLG = 1;
end
% End NNL %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Statistics %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if ~isempty(Stats)
    if ischar(Stats)
       StatsNames = {'on','off'};
       jst = strcmpi(Stats,StatsNames);
           if sum(jst) == 0
              error(['Invalid value for the ''Statistics'' parameter: '...
                     'choices are ''on'' or ''off''.']);
           end
       Stats = StatsNames{jst}; 
    else  
        error(['Invalid value for the ''Statistics'' parameter: '...
               'choices are ''on'' or ''off''.']);
    end
end
StatsFLG = 0;
jS = strcmpi(Stats,'on');
if sum(jS) == 1
       StatsFLG = 1;
end
% End Statistics %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Normalization %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if ~isempty(normalization)
    if ischar(normalization)
       NormNames = {'off','standard','minmax'};
       jnl = strcmpi(normalization,NormNames);
           if sum(jnl) == 0
              error(['Invalid value for the ''normalization'' parameter: '...
                     'choices are ''off'' or ''standard'' or ''minmax''.']);
           end
       normalization = NormNames{jnl}; 
       switch normalization
       case 'off'
           Xs = X-mean(X);            
       case 'standard'
           Xs = zscore(X,1);
       case 'minmax'
           un = ones(n,1);
           minX = min(X);
           maxX = max(X);
           Xs = (X-un*minX)./(un*(maxX - minX));
           Xs = Xs-mean(Xs);
       end
    else  
        error(['Invalid value for the ''normalization'' parameter: '...
                     'choices are ''off'' or ''standard'' or ''minmax''.']);
    end
else
     error(['Invalid value for the ''normalization'' parameter: '...
                     'choices are ''off'' or ''standard'' or ''minmax''.']);
end
% End Normalization %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Significance Level %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if ~isempty(alpha)
    if isnumeric(alpha)
        if alpha >= 1 || alpha <= 0
            error(['Invalid value for alpha'...
                'choices must be in (0,1).']);
        end
    end
end
% End Significance Level %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Start the algorithm %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
flg = 1;

while flg == 1
      if NNLFLG == 1
          if sum(constrV) ~= 0
              [V,A,Y,~,~] = DPCA(Xs,Q,'Stand','off','Rndst',mlst,'Constr',constrV,'NN','on');                       
          else
              [V,A,Y,~,~] = DPCA(Xs,Q,'Stand','off','Rndst',mlst,'NN','on');   
          end
      else
          if sum(constrV) ~= 0
              [V,A,Y,~,~] = DPCA(Xs,Q,'Stand','off','Rndst',mlst,'Constr',constrV);
          else
              [V,A,Y,~,~] = DPCA(Xs,Q,'Stand','off','Rndst',mlst);       
          end
      end
    if sum(V,2) ~= 0
        flg = 0;
    end
end

A_merge{Q} = A; 
Y_merge{Q} = Y;
V_merge{Q} = V; 
Yb = Y;
Ab = A;

Ym = Xs*A;
Xt = Ym*A'; 
of{Q} = trace((Xs-Xt)'*(Xs-Xt));

% Model Assessment: DPCA %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
R2SCI(Q) = trace((Y*A')'*(Y*A'))/trace(Xs'*Xs);
R2SCIh{Q} = zeros(Q,1);
for h = 1:Q
    Jh = V(:,h);
    JCh = [vecJ(Jh==1)];
    R2SCIh{Q}(h) = trace((Y(:,h)*(A(:,h))')'*(Y(:,h)*(A(:,h))'))/trace((Xs(:,JCh))'*Xs(:,JCh));
    cro{Q}(h) = CronbachAlpha(Xs(:,JCh));
end   
expvar(Q)= trace(Y'*Y)./trace(Xs'*Xs)*100;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

PQ = V*vecQ;

for q = 1:Q
    ucq = PQ(q);
    if ucq > q
        vq = PQ == q;
        vpq = PQ == ucq;
        PQ(vq) = ucq;
        PQ(vpq) = q;
    end
    vq = find(PQ == q);
    lbV(q) = min(vq);
end

for istep = Q-1:-1:1
    % Identity test on the component correlation matrix (Chen,Zhang,Zhong, 2010).
    pval = idtest(Y_merge{istep+1});
    
    % Reflective model if the p-value is lower than the nominal level.
    if pval < alpha 
        
        test = 1;
        oft = of{istep+1};
        Y_m = zeros(n,istep);
        A_m = zeros(J,istep);
        of1 = Inf;
        colsagg = zeros(J,1);
        
        if istep == 1 && ACPFLG == 0 
            [gCI,cCI] = CompositeInd(Y_merge{istep+1},NNLFLG);
            CI = gCI;
            w  = cCI;
            Y_merge{istep} = gCI;
            A_merge{istep} = w;
            V_merge{istep} = ones(J,1);
            AV(istep) = find(any(V),1,'first');
            BV(istep) = find(any(V),1,'last');
            of{istep} = trace((Y_merge{istep+1}-Y_merge{istep}*A_merge{istep}')'*(Y_merge{istep+1}-Y_merge{istep}*A_merge{istep}'))+of{istep+1};
            levfusaV(istep) = of{istep};
            % Model Assessment %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            R2SCI(istep) = trace((Y_merge{istep}*A_merge{istep}')'*(Y_merge{istep}*A_merge{istep}'))/trace(Y_merge{istep+1}'*Y_merge{istep+1});
            cro{istep} = CronbachAlpha(Xs);
            expvar(istep)= trace((Y_merge{istep}*A_merge{istep}')'*(Y_merge{istep}*A_merge{istep}'))./trace(Y_merge{istep+1}'*Y_merge{istep+1})*100;
            disp(sprintf('\n \n For the last equation of reflective model connected to the hierarchy (Y2 = g*V1^t*B1+E1): \n R2g=%g, Explained variance=%g, Cronbach alpha=%g',R2SCI(istep),expvar(istep),cro{istep})) 
            disp(sprintf('\n No monotonocity is expected with respect to the hierarchy. \n The last level of the model, i.e. q = 1, reconstructs 2 components instead of the original matrix.'))
            break
        else
            for p = 1:Q-1
                if PPV(p) == p
                    for q = p+1:Q
                        if PPV(q) == q
                            iQ = vecQ ~= p & vecQ ~= q;
                            vpq = V(:,p)+V(:,q);
                            V_m = [V(:,iQ) vpq];
                            V_m = V_m(:,any(V_m)~=0);
                            JCg = [vecJ(vpq==1)];
                            [a,y,~] = ACP(Xs(:,JCg),NNLFLG);
                            A_mm = a(:,1);
                            Y_mm = y(:,1);
                            AA = zeros(J,1);
                            AA(JCg,1) = A_mm;
                            A_m = [Ab(:,iQ) AA];
                            A_m = A_m(:,any(A_m)~=0);
                            Y_m = [Yb(:,iQ) Y_mm];
                            Y_m = Y_m(:,any(Y_m)~=0);
                            
                            Xt_m = Y_m*A_m';
                            E = Xs-Xt_m;
                            
                            ofm = trace(E'*E);
                            
                            if ofm < of1
                                
                                ica = p;
                                jca = q;
                                of1 = ofm;
                                colsagg = vpq;
                                loadagg = AA;
                                scoreagg = Y_mm;
                                Y_merge{istep} = Y_m;
                                A_merge{istep} = A_m;
                                V_merge{istep} = V_m;
                            end
                            Y_m = zeros(n,istep);
                            A_m = zeros(J,istep);
                        end
                    end
                end
            end
            
            V(:,ica) = colsagg;
            V(:,jca) = 0;
            
            Ab(:,ica) = loadagg;
            Ab(:,jca) = 0;
            
            Yb(:,ica) = scoreagg;
            Yb(:,jca) = 0;
            
            of{istep} = of1+of{istep+1};
            levfusaV(istep) = of{istep};
            
            for j = 1:Q
                if PPV(j) == jca
                    PPV(j) = ica;
                end
            end
            
            lbV(ica) = min(lbV(ica),lbV(jca));
            lbV(jca) = max(lbV(ica),lbV(jca));
            AaV(istep) = lbV(ica);
            BaV(istep) = lbV(jca);
            
            AV(istep) = ica;
            BV(istep) = jca;
      
            % Model Assessment %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            R2SCI(istep) = trace((Y_merge{istep}*A_merge{istep}')'*(Y_merge{istep}*A_merge{istep}'))/trace(Xs'*Xs);
            R2SCIh{istep} = zeros(istep,1);
            for h = 1:istep
                Jh = V_merge{istep}(:,h);
                JCh = [vecJ(Jh==1)];
                R2SCIh{istep}(h) = trace((Y_merge{istep}(:,h)*(A_merge{istep}(:,h))')'*(Y_merge{istep}(:,h)*(A_merge{istep}(:,h))'))/trace((Xs(:,JCh))'*Xs(:,JCh));
                cro{istep}(h) = CronbachAlpha(Xs(:,JCh));
            end
            expvar(istep)= trace(Y_merge{istep}'*Y_merge{istep})./trace(Xs'*Xs)*100;
            
            % Path statistics for variables %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            ass = zeros(J,1);
            p = zeros(J,1);
            Corrvec = zeros(J,1);
            for j=1:J
                for q = 1:istep
                    if A_merge{istep}(j,q)~=0
                        p(j)= r2pv(corr(Xs(:,j),Y_merge{istep}(:,q)),n);
                        Corrvec(j) = corr(Xs(:,j),Y_merge{istep}(:,q));
                        ass(j) = q;
                    end
                end
            end
            
            Pathtab = table(vecJ,ass,Corrvec,p);
            Pathmat{istep} = table2cell(Pathtab);
            
            if istep > 1 &  StatsFLG == 1
                % Path statistics for Factors %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                disp(sprintf('\n \n Factor1         Factor2         Correlation       pvalue'))
                ppf = [1:istep-1]';
                assf1 = zeros(sum(ppf),1);
                assf2 = zeros(sum(ppf),1);
                pf = zeros(sum(ppf),1);
                Corrvecf = zeros(sum(ppf),1);
                loop = 1;
                for q = 1:istep-1
                    for qq = q+1:istep
                        assf1(loop) = q;
                        assf2(loop) = qq;
                        Corrvecf = corr(Y_merge{istep}(:,q),Y_merge{istep}(:,qq));
                        pf = r2pv(corr(Y_merge{istep}(:,q),Y_merge{istep}(:,qq)),n);
                        loop = loop+1;
                        disp(sprintf('%f        %f         %f         %f',q,qq,Corrvecf,pf))
                    end
                end
                % End Path Statistics for Factors %%%%%%%%%%%%%%%%%%%%%%%%%
                
                clear nn tcomp;
            end
        end
    else
        stopformative = 1;
        llstep = istep+1;
        lclus = find(PPV == [1:Q]');
        break
    end
    if istep == 1
        disp(sprintf('\n \n Last level of the reflective model: X = Y1*V1^t*B1 + E1.'))
    end
end

if test == 0
    llstep = Q;
    lclus = find(PPV == [1:Q]');
end

% Formative part of the model %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if stopformative == 1
    [CI,w] = CompositeInd(Y_merge{llstep},NNLFLG,'MeasureMod','Formative','DataMatrix',Xs);
    for tstep = llstep-1:-1:1
        AV(tstep) = min(lclus);
        BV(tstep) = lclus(size(lclus,1)-tstep+1);
        if test == 1
            levfusaV(tstep) = levfusaV(llstep)+1000;
        else
            levfusaV(tstep) = of{Q}+1000;
        end
    end
    M = llstep;
else
    M = 1;
    if ACPFLG == 1
        CI = Y_merge{1};
        w = A_merge{1};
    end
end
% End Formative %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

hierV = table(AV,BV,levfusaV');
hierV.Properties.VariableNames = {'AV' 'BV' 'levfusaV'};

% DENDROGRAM of V %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
levfusaV = levfusaV';
ZV = [flipud(AV) flipud(BV) flipud(levfusaV)];
% figure();
 [~,~,outV] = dendrogram(ZV,Q);
% title({'Hierarchical Disjoint Principal Component Analysis'})
% xlabel({'Components'})
% ylabel({'Objective Function'})


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Graph Representation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

nnode = (Q*2)-1+J;
fnode = Q+J;

class = V_merge{Q}*[1:Q]';
%[S,I] = sort(class);
s = zeros(1,J);
I = zeros(1,J);
sizesq = 0;
c = 1;
for q = 1:Q
    qq = outV(q);
    sq = find(class == qq);
    s(sizesq+1:sizesq+size(sq,1)) = qq;
    t(sizesq+1:sizesq+size(sq,1)) = c;
    I(sizesq+1:sizesq+size(sq,1)) = sq;
    sizesq = sizesq+size(sq,1);
    c = c+1;
end

c = 1;
tabd = [AV BV [fnode+1+Q-2:-1:fnode+1]'];
tabnode = zeros(Q-1,2);
mintab(c,1) = min(tabd(Q-1,1),tabd(Q-1,2));
mintab(c,2) = tabd(Q-1,3);
tabnode(Q-1,:) = tabd(Q-1,1:2);
for q = Q-2:-1:1
    tr = 0;
    ft1 = find(tabd(q,1) == mintab(:,1));
    ft2 = find(tabd(q,2) == mintab(:,1));
    if ~isempty(ft2)
        tabnode(q,2) = mintab(ft2,2); 
        mintab(ft2,2) = mintab(ft2,2);
        tr = tr+1;
    end
        if ~isempty(ft1)
        tabnode(q,1) = mintab(ft1,2);
        mintab(ft1,2) = max(mintab(ft1,2),tabd(q,3));
        tr = tr+1;
        end
    if tr == 1 
        if isempty(ft2)
            tabnode(q,2) = tabd(q,2);
        elseif isempty(ft1)
            tabnode(q,1) = tabd(q,1);
            c = c+1;
            mintab(c,1) = tabd(q,1);
            mintab(c,2) = tabd(q,3);
        end
    end
    if tr == 0 
        c = c+1;
        mintab(c,1) = min(tabd(q,1),tabd(q,2));
        mintab(c,2) = tabd(q,3);
        tabnode(q,:) = tabd(q,1:2);
    end
end


newclus = [[1:Q]' outV'];

tabnode1=zeros(Q-1,2);
for qq=1:Q
    k=find(tabnode==outV(qq));
    tabnode1(k)=qq;
end
for qq= 1:Q-1
    for qqq=1:2
        if tabnode1(qq,qqq)==0
            tabnode1(qq,qqq)=tabnode(qq,qqq);
        end
    end
end

tt = [Q+1:fnode];

% Nodes Size: Variance of the Components (problems with non-standardised
% matrices) or Cronbach's Alpha.
%SizeNodes = [(floor(diag((1/n)*Y_merge{Q}(outV)'*Y_merge{Q}(outV)))+1)' ones(1,J)];
sQnode = cro{Q}(outV);
sQnode(find((cro{Q}(outV)<0.7 | isnan(cro{Q}(outV)))==1)) = 2;
sQnode(find((cro{Q}(outV)>0.7)==1)) = 20.*sQnode(find((cro{Q}(outV)>0.7)==1));
SizeNodes = [sQnode ones(1,J)];
LabelNodes = [outV I];
G = digraph(t,tt,[],fnode);
figure();
H = plot(G,'XData',[(J/(Q+1))*[1:Q] [1:J]], 'YData',[(log(levfusaV(Q-1)/2))*ones(1,Q) (log(levfusaV(Q-1)/3))*ones(1,J)],'ArrowSize',11,'ArrowPosition',0.8,'MarkerSize',SizeNodes,'LineWidth',0.9);
H.NodeFontSize = 12;
x0data = [[1:Q]' ((J/(Q+1))*[1:Q])'];
%Add node from Q to 1
d = 1;
xxdata = zeros(Q-1,1);
ydata = zeros(Q-1,1);
for q = Q-1:-1:M
    %varYn(d) = (1/n)*Y_merge{q}(:,q)'*Y_merge{q}(:,q);
    %SizeNodes = [(floor(diag((1/n)*Y_merge{Q}'*Y_merge{Q}))+1)' ones(1,J) varYn(1:d)];
    if cro{q}(q) < 0.7
        snode(d) = 1;
    else
        snode(d) = 10.*cro{q}(q);
    end
    SizeNodes = [sQnode ones(1,J) snode(1:d)];
    ln(d) = Q+d;
    LabelNodes = [outV I ln(1:d)];
    G = addnode(G,1);
    G = addedge(G,[Q+J+d Q+J+d],[tabnode1(q,1) tabnode1(q,2)]);
    p = find(x0data(:,1)==tabnode1(q,1));
    pp = find(x0data(:,1)==tabnode1(q,2));
    xxdata(d) = (x0data(p,2)+x0data(pp,2))/2;
    ydata(d) = log(levfusaV(q));
    H = plot(G,'XData',[(J/(Q+1))*[1:Q] [1:J] xxdata(1:d)'], 'YData',[(log(levfusaV(Q-1)/2))*ones(1,Q) (log(levfusaV(Q-1)/3))*ones(1,J) ydata(1:d)'],'ArrowSize',7,'ArrowPosition',0.8,'MarkerSize',SizeNodes,'NodeLabel',LabelNodes,'LineWidth',0.9);
    H.NodeFontSize = 12;
    hold on
    highlight(H,[Q+J+d Q+J+d],[tabnode1(q,1) tabnode1(q,2)],'EdgeColor',[0.5176 0 0])
    if q == M
        title({'Path Diagram of HDPCA'})
        xlabel({'Components'})
        ylabel({'Log(objective function)'})
    end
    x0data(Q+d,1) = Q+J+d;
    x0data(Q+d,2) = xxdata(d);
    d = d+1;
end

if M > 1
    G = addnode(G,1);
    if test == 1
        ydata(d) = log(levfusaV(q))+2;
        %SizeNodes = [(floor(diag((1/n)*Y_merge{Q}'*Y_merge{Q}))+1)' ones(1,J) varYn(1:d-1) (varYn(d-1)+1)];
        if CronbachAlpha(Y_merge{M}) < 0.7
            sMnode = 2;
        else
            sMnode = 10.*CronbachAlpha(Y_merge{M});
        end
        SizeNodes = [sQnode ones(1,J) snode(1:d-1) sMnode];
    else
        ydata(d) = log(of{Q})+2;
        %varYn(d) = (1/n)*CI'*CI;
        %SizeNodes = [(floor(diag((1/n)*Y_merge{Q}'*Y_merge{Q}))+1)' ones(1,J) varYn(d)];
        if CronbachAlpha(Xs) < 0.7
            sMnode = 2;
        else
            sMnode = 10.*CronbachAlpha(Xs);
        end
        SizeNodes = [sQnode ones(1,J) sMnode];
    end
    nodeM = Q+J+d*ones(1,M);
    archM = [tabnode1(M-1,1) tabnode1(M-1,2)  tabnode1([1:M-2]',2)'];
    G = addedge(G,archM,nodeM);
    for m = M:-1:1
        p(m) = find(x0data(:,1)==archM(m));
    end
    xxdata(d) = mean(x0data(p,2));
    ln(d) = Q+d;
    LabelNodes = [outV I ln(1:d)];
    H = plot(G,'XData',[(J/(Q+1))*[1:Q] [1:J] xxdata(1:d)'], 'YData',[(log(levfusaV(Q-1)/2))*ones(1,Q) (log(levfusaV(Q-1)/3))*ones(1,J) ydata(1:d)'],'ArrowSize',7,'ArrowPosition',0.8,'MarkerSize',SizeNodes,'NodeLabel',LabelNodes,'LineWidth',0.9);
    H.NodeFontSize = 12;
    hold on
    highlight(H,archM,nodeM,'EdgeColor',[0.8745 0.4627 0])
    labelnode(H,tt,I)
    title({'Path Diagram of HDPCA'})
    xlabel({'Components'})
    ylabel({'Log(objective function)'})
end

% ------------------------ Local Functions---------------------------------

function [a,y,itr] = ACP(Xr,NNLFLG)
% ACP function with the power method.

maxit=300;
[n,Q]=size(Xr);
a = rand(Q,1); 
tol = 1e-6;
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


function [as,varargout] = CronbachAlpha(x)
% CronbachAlpha
% Description:	calculate Cronbach's alpha for a set of psychometric
%               measurements.
% Syntax:	[as,au] = CronbachAlpha(x)
% In:
% 	x	- an nRep x nItem array of ratings, so that each row is the set of
%		  obvservations from one repetition and each column is the set of all
%		  observations for a given item
% Out:
% 	as	- the standardized Cronbach's alpha
%	au	- the unstandardized Cronbach's alpha
% Updated: 2012-09-24
% Copyright 2012 Alex Schlegel (schlegel@gmail.com).  This work is licensed
% under a Creative Commons Attribution-NonCommercial-ShareAlike 3.0 Unported
% License.
nItem = size(x,2);
%logical array for selecting upper triangular part of the correlation and
%covariance matrices, where the good stuff is
	b = triu(true(nItem),1);
%standardized alpha
	%pairwise correlations between items
		r = corrcoef(x);
	%mean of the meaningful, non-redundant correlations
		r = mean(r(b));
	as = nItem*r/(1 + (nItem-1)*r);
%unstandardized alpha
if nargout>1
	%variance/covariance matrix
		vc = cov(x);
	%mean variance (variances are along the diagonal)
		v = mean(diag(vc));
	%mean covariance, not including variances
		c = mean(vc(b));
	varargout{1} = nItem*c/(v + (nItem-1)*c);
end

function p = r2pv(r,n)
%
% p = r2pv(r,n)
%
% r = Estimated correlation coefficient (IE |r| <= 1)
%   = (1/n)*(x'*y) for vectors (x,y) of length n
% n = no. samples used
% p = P-value based on |r| (two sided) with rho=0 (null case)
%
% NOTES: following Cramer, p.400, convert r to a t and use what we have for t. 
if n < 3
    error('n < 3');
end
if r==1. 
    p=0; 
    return;
end
%t=sqrt(n-2)*r/(sqrt(1-r*r)); 	% This is t with n-2 d.f.
t=r*sqrt((n-2)/(1-r*r));
t=abs(t);						% Use |t| for two sided P-value
p=2*(1-tcdf(t,n-2));







