%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Testing identity of high-dimensional covariance matrix (Chen,Zhang,Zhong)%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function p = idtest(X)
[n,J] = size(X);
X = zscore(X,1);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Y1n = (1./n)*trace(X'*X);

Y3n = (1./(n*(n-1)))*(sum(sum(X*X'))-trace(X*X'));

T1n = Y1n - Y3n;

Y2 = sum(sum((X*X').^2))-trace((X*X').^2);
Y2n = (1./(n*(n-1)))*(sum(sum((X*X').^2))-trace((X*X').^2));

XX = X*X'-diag(diag(X*X'));
Y4 = sum(sum(XX^2))-trace(XX^2);

Y4n = (1./(n*(n-1)*(n-2)))*(sum(sum(XX^2))-trace(XX^2));

Y9 = trace((X*X').^2);
Y10 = sum(sum(X*X'*diag(diag(X*X'))));
Y13 = sum(sum(X*X'*X*X'));
Y8 = Y13-2*Y10+Y9;

Y7 = sum(sum(kron(X*X',X*X')))-2*Y13-sum(sum(kron(diag(diag(X*X')),X*X')))-sum(sum(kron(X*X',X.^2)))+2*Y10-2*Y9+sum(sum(kron(diag(diag(X*X')),diag(diag(X*X')))));
Y5 = Y7-Y8+Y2-Y4;

Y5n = (1./(n*(n-1)*(n-2)*(n-3)))*Y5;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Test Statistic
T2n = Y2n-2*Y4n+Y5n;

Vn = (1./J)*T2n - (2./J)*T1n + 1;

TestN = (n/2)*Vn;

p = 2*(1-normcdf(abs(TestN)));
