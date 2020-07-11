pkg load statistics

% Partiendo de dos variables aleatorias de distribucion uniforme,
% se busca la densidad de probabilidad conjunta y la funcion de probabilidad

% Siendo variables independientes identicamente distribuidas y dado que pdf(x)=
% pdf(y)=1=> pdf(x,y)=1

N=100;
x=linspace(0,1,N);
y=linspace(0,1,N);
z=ones(N,N); %pdf(x,y)=1

figure(1)
plot_pdf=surf(x,y,z)
xlabel('var aleatoria x')
ylabel('var aleatoria y')
zlabel('pdf conjunta')
title('Pdf conjunta de dos variable aleatorias uniformes')

% Si a la pdf conjunta pdf(x,y)=1 le calculamos la integral de -inf a inf,
% obtenemos la cdf(x,y)=x*y
cdf=zeros(N,N);

for i=1:length(x)
  for j=1:length(y)
    cdf(i,j)=x(i)*y(j);
  endfor
endfor

figure(2)
plot_cdf=surf(x,y,cdf)
xlabel('var aleatoria x')
ylabel('var aleatoria y')
zlabel('cdf conjunta')
title('cdf conjunta de dos variable aleatorias uniformes')

% Dado que son variables independientes P(X>0.7,Y<0.4)=P(X>0.7)*P(Y<0.4)
p_x1=1-unifcdf(0.7);
p_y1=unifcdf(0.4);
p1=p_x1*p_y1;

% Para el percentil 40 de X se debe encontrar la cdf marginal de x, que en 
% este caso es lim y->inf cdf(x,y). y->inf cdfy(y)=1 => cdf(0.4,inf)=cdfx(0.4)
p_40=unifcdf(0.4);