pkg load statistics

mu=0;
sigma=1;

%Probabilidad de X>0.7 e Y<0.4-->nuevamente al ser variables independientes
%podemos tomar las cdf de X e Y y multiplicar las probabilidades

px_gt_07=1-mvncdf(0.7,mu,sigma);
py_lt_04=mvncdf(0.4,mu,sigma);
pxy=px_gt_07*py_lt_04

%Grafica de pdf conjunta segun https://github.com/ejesposito/ceai/blob/master/Prob_Est/Clase1/ej4.m

mu = [0 0];
sigma = [1 0; 0 1];

x = -3:0.2:3;
y = -3:0.2:3;
[X,Y] = meshgrid(x,y);
XY = [X(:) Y(:)];

% pdf conjunta
pdf = mvnpdf(XY,mu,sigma);
pdf = reshape(pdf,length(Y),length(X));

figure(1)
h = surf(X,Y,pdf);
caxis([min(pdf(:))-0.5*range(pdf(:)),max(pdf(:))])
xlabel('X')
ylabel('Y')
zlabel('PDF CONJUNTA')

