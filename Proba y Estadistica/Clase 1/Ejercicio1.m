pkg load statistics

%Probabilidad de obtener 3 cecas en 10 tiradas
prob_3cecas_eq=nchoosek(10,3)*(0.5^3)*((1-0.5)^7)
prob_3cecas_eq_pkg=binopdf(3,10,0.5)

%Probabilidad de obtener 3 cecas en 10 tiradas en moneda desequilibrada P(ceca)=0.4
prob_3cecas=nchoosek(10,3)*(0.4^3)*((1-0.4)^7)
prob_3cecas_pkg=binopdf(3,10,0.4)

%Probabilidad de dado cara, obtener al menos 3 cecas en 10 tiradas en la moneda desequilibrada
%El enfoque aquí es inverso, calcular la probabilidad de que salgan 8, 9 o 10 cecas
prob_gteq3cecas=1-(nchoosek(10,8)*(0.6^8)*((1-0.6)^2)+nchoosek(10,9)*(0.6^9)*((1-0.6)^1)+nchoosek(10,10)*(0.6^10)*((1-0.6)^0))
prob_gteq3cecas_pkg=1-binocdf(2,10,0.4)
