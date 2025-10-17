% Poner el directorio pertinente
cd /home/

DatMPGOct=csvread('mpg.csv');
DatMPGOct(1:6,:)

DatMPGOct=csvread('mpgOct.csv');
DatMPGOct(1:6,:)

hwyOct=DatMPGOct(:,1);
drvOct=DatMPGOct(:,2);
cylOct=DatMPGOct(:,3);
manufOct=DatMPGOct(:,4);
displOct=DatMPGOct(:,5);
hwyOct(1:6)'

N=length(hwyOct);
figure 1
% El primer vector describe la coordenada x 
% (en este caso la variable hwy)
% El segundo vector describe la coordenada y 
% (en este caso un vector de unos con 1 fila y N columnas)
% El tercer argumento permite graficar las observaciones como puntos, 
% por default los une con líneas según el orden de aparición
plot(hwyOct,ones(1,N),'o')
hold on
% hold on permite hacer modificaciones en el gráfico antes de terminar la figura
% en este caso, le cambiamos el nombre al eje x
xlabel('Rendimiento en autopista [millas/galón]')
hold off

figure 2
% En este caso, no agrego un tercer parámetro ya que el default une 
% las observaciones con líneas
% según el orden del vector. Para visualizar el orden, en el eje x se pone un
% vector que va de 1 a N.
plot(1:N,hwyOct)
hold on
xlabel('Número de observación')
ylabel('Rendimiento en autopista [millas/galón]')
hold off

muHWY=sum(hwyOct)/N
RmuHWY=mean(hwyOct)

sdHWY=sqrt(sum((hwyOct-muHWY).^2)/(N-1))
RsHWY=std(hwyOct)

gHWY=sum((hwyOct-muHWY).^3)/(N*sdHWY^3)
RgHWY=skewness(hwyOct)

kHWY=sum((hwyOct-muHWY).^4)/(N*sdHWY^4)-3
RkHWY=kurtosis(hwyOct)
RkHWY=kurtosis(hwyOct)-3

sHWYOct=sort(hwyOct);
hwyOct(1:6)'
sHWYOct(1:6)'

MinHWYOct=sHWYOct(1)
RMinHWYOct=min(hwyOct)

MaxHWYOct=sHWYOct(N)
RMaxHWYOct=max(hwyOct)

ParteEntera2=floor(N/2);
Paridad=floor(N/2)==N/2;
if Paridad==1
  % En caso de paridad, tomo el promedio
  MedHWYOct=(sHWYOct(ParteEntera2)+sHWYOct(ParteEntera2+1))/2;
else
  % Sino, tomo el valor del medio
  MedHWYOct=sHWYOct(ParteEntera2+1);
end
MedHWYOct
RMedHWYOct=median(hwyOct)


ParteEntera4=floor(N/4);
DivX4=ParteEntera4==N/4;
if DivX4==1
  % Si es divisible por 4 tomo el promedio de los valores sucesivos
  Q1HWYOct=(sHWYOct(ParteEntera4)+sHWYOct(ParteEntera4+1))/2;
else
  % Si no es divisible por 4, tomo el valor siguiente
  Q1HWYOct=sHWYOct(ParteEntera4+1);
end
Q1HWYOct
RQ1HWYOct=quantile(hwyOct,0.25)

if DivX4==1
  % Si es divisible por 4 tomo el promedio de los valores sucesivos
  Q3HWYOct=(sHWYOct(3*ParteEntera4)+sHWYOct(3*ParteEntera4+1))/2;
else
  % Si no es divisible por 4, tomo el valor siguiente
  Q3HWYOct=sHWYOct(3*ParteEntera4+1);
end
Q3HWYOct
RQ3HWYOct=quantile(hwyOct,0.75)

ParteEntera20=floor(N*5/100);
DivX20=ParteEntera20==N*5/100;
if DivX20==1
  % Si es divisible por 20 tomo el promedio de los valores sucesivos
  P5HWYOct=(sHWYOct(ParteEntera20)+sHWYOct(ParteEntera20+1))/2;
else
  % Si no es divisible por 20, tomo el valor siguiente
  P5HWYOct=sHWYOct(ParteEntera20+1);
end
P5HWYOct
RP5HWYOct=quantile(hwyOct,0.05)

boxplot(hwyOct)

% Debo cargar el paquete statistics
pkg load statistics
boxplot(hwyOct)

boxplot(hwyOct);
hold on
% Cambio los ejes para que quede centrado el boxplot
xlim([0.4 1.6])
ylabel('Rendimiento en autopista [millas/galón]')
hold off

% IndDRV1 devuelve las coordenadas en las que la variable drv es 1
IndDRV1=find(drvOct==1);
% IndDRV2 devuelve las coordenadas en las que la variable drv es 2
IndDRV2=find(drvOct==2);
% IndDRV3 devuelve las coordenadas en las que la variable drv es 3
IndDRV3=find(drvOct==3);
% Se ve que seleccionando filas de IndDRV1, la segunda columna 
% de la base tiene valor 1 como constante
DatMPGOct(IndDRV1(1:6),:)
% Se ve que seleccionando filas de IndDRV2, la segunda columna 
% de la base tiene valor 2 como constante
DatMPGOct(IndDRV2(1:6),:)
% Se ve que seleccionando filas de IndDRV3, la segunda columna 
% de la base tiene valor 3 como constante
DatMPGOct(IndDRV3(1:6),:)
% Selecciono los rendimientos en autopista para cada valor de la variable drv
hwyDRV1=hwyOct(IndDRV1);
hwyDRV2=hwyOct(IndDRV2);
hwyDRV3=hwyOct(IndDRV3);

% Para hacer boxplots paralelos deben incluirse los vectores 
% de distinta longitud en lo que se llama una célula
% denotado con llaves
boxplot ({hwyDRV1,hwyDRV2,hwyDRV3});
hold on
xlim ([0,4])
% set(gca()) permite cambiar características gráficas preestablecidas
% en este caso, el eje x debe expresarse con caracteres.
set(gca (), "xtick", [1 2 3], "xticklabel",{"4x4", "Dirección delantera", "Dirección trasera"})
ylabel('Rendimiento en autopista [millas/galón]')
hold off

hist(hwyOct,20)
hold on
xlabel('Rendimiento en autopista')
ylabel('Frecuencia absoluta')
hold off

hist(hwyOct,12:2:44)
hold on
xlabel('Rendimiento en autopista')
ylabel('Frecuencia absoluta')
hold off

[Hc Hm]=hist(hwyOct,12:2:44);
Hc
Hm

bar(Hm,Hc)
hold on
xlabel('Rendimiento en autopista')
ylabel('Frecuencia absoluta')
hold off


bar(Hm,Hc,1)
hold on
xlabel('Rendimiento en autopista')
ylabel('Frecuencia absoluta')
hold off

bar(Hm,Hc,1)
hold on
plot([11 Hm 45],[0 Hc 0],'-or')
xlabel('Rendimiento en autopista')
ylabel('Frecuencia absoluta')
hold off

bar(Hm,Hc/N,1)
hold on
plot([11 Hm 45],[0 Hc/N 0],'-or')
xlabel('Rendimiento en autopista')
ylabel('Frecuencia relativa')
hold off

bar(Hm,cumsum(Hc),1)
hold on
plot([Hm-1 44],[0 cumsum(Hc)],'-or')
xlabel('Rendimiento en autopista')
ylabel('Frecuencia absoluta acumulada')
hold off

bar(Hm,cumsum(Hc)/N,1)
hold on
plot([Hm-1 44],[0 cumsum(Hc)/N],'-or')
xlabel('Rendimiento en autopista')
ylabel('Frecuencia relativa acumulada')
hold off

muAgOct=sum(Hm.*Hc)/N

DifxiMuAgCuad=(Hm-muAgOct).^2;
sdAgOct=sqrt(sum(Hc.*DifxiMuAgCuad)/(N-1));

DifxiMuAgCub=(Hm-muAgOct).^3;
gAgOct=sum(Hc.*DifxiMuAgCub)/(N*sdAgOct^3)

DifxiMuAgCuar=(Hm-muAgOct).^4;
kAgOct=sum(Hc.*DifxiMuAgCuar)/(N*sdAgOct^4)-3


function pKAg=kPercAgOct(k,f,x)
  N=sum(f);
  F=cumsum(f);
  bw=diff(x);
  bw=bw(1);
  LI=x-bw/2;
  DatAc=N*k/100;
  Enc=0;
  i=0;
  while(Enc==0)
    i=i+1;
    Enc=F(i)>DatAc;
  end
  pKAg=LI(i)+ (DatAc-F(i-1))* bw/f(i);
end

kPercAgOct(50,Hc,Hm)
kPercAgOct(25,Hc,Hm)
kPercAgOct(75,Hc,Hm)
kPercAgOct(5,Hc,Hm)

subplot(311)
hist(hwyDRV1,20)
title('4x4')
xlabel('Rendimiento en autopista [millas/galón]')
ylabel('Frecuencia Absoluta')
subplot(312)
hist(hwyDRV2,20)
title('Direccion delantera')
xlabel('Rendimiento en autopista [millas/galón]')
ylabel('Frecuencia Absoluta')
subplot(313)
hist(hwyDRV3,20)
title('Direccion trasera')
xlabel('Rendimiento en autopista [millas/galón]')
ylabel('Frecuencia Absoluta')

subplot(311)
hist(hwyDRV1,20)
title('4x4')
xlabel('Rendimiento en autopista [millas/galón]')
ylabel('Frecuencia Absoluta')
xlim([10 46])
ylim([0 30])
subplot(312)
hist(hwyDRV2,20)
title('Direccion delantera')
xlabel('Rendimiento en autopista [millas/galón]')
ylabel('Frecuencia Absoluta')
xlim([10 46])
ylim([0 30])
subplot(313)
hist(hwyDRV3,20)
title('Direccion trasera')
xlabel('Rendimiento en autopista [millas/galón]')
ylabel('Frecuencia Absoluta')
xlim([10 46])
ylim([0 30])


m1=mean(hwyDRV1);
m2=mean(hwyDRV2);
m3=mean(hwyDRV3);
med1=median(hwyDRV1);
med2=median(hwyDRV2);
med3=median(hwyDRV3);


subplot(311)
hist(hwyDRV1,20)
hold on
title('4x4')
xlabel('Rendimiento en autopista [millas/galón]')
ylabel('Frecuencia Absoluta')
xlim([10 46])
ylim([0 30])
plot([m1 m1],[0 30],'--r')
plot([med1 med1],[0 30],':g')
hold off
subplot(312)
hist(hwyDRV2,20)
hold on
title('Direccion delantera')
xlabel('Rendimiento en autopista [millas/galón]')
ylabel('Frecuencia Absoluta')
xlim([10 46])
ylim([0 30])
plot([m2 m2],[0 30],'--r')
plot([med2 med2],[0 30],':g')
hold off
subplot(313)
hist(hwyDRV3,20)
hold on
title('Direccion trasera')
xlabel('Rendimiento en autopista [millas/galón]')
ylabel('Frecuencia Absoluta')
xlim([10 46])
ylim([0 30])
plot([m3 m3],[0 30],'--r')
plot([med3 med3],[0 30],':g')
hold off

[F X]=hist(cylOct,4:8)
bar(X,F)
xlabel('Cantidad de cilindros')
ylabel('Frecuencia absoluta')
stem(X,F)
xlabel('Cantidad de cilindros')
ylabel('Frecuencia absoluta')
xlim([3 8])

plot(displOct,hwyOct,'o')
hold on
xlabel('Desplazamiento del motor [litros]')
ylabel('Rendimiento en autopista [millas/galón]')
hold off



Rcta=polyfit(displOct,hwyOct,1);
x1=1.5;
x2=6.5;
y1=polyval(Rcta,x1);
y2=polyval(Rcta,x2);
plot(displOct,hwyOct,'o')
hold on
plot([x1 x2],[y1 y2],'r')
xlabel('Desplazamiento del motor [litros]')
ylabel('Rendimiento en autopista [millas/galón]')
hold off

sqDisplOct=sqrt(displOct);
Rcta=polyfit(sqDisplOct,hwyOct,1);
x1=sqrt(1.5);
x2=sqrt(6.5);
y1=polyval(Rcta,x1);
y2=polyval(Rcta,x2);
plot(sqDisplOct,hwyOct,'o')
hold on
plot([x1 x2],[y1 y2],'r')
xlabel('Raíz(Desplazamiento del motor [litros]) ')
ylabel('Rendimiento en autopista [millas/galón]')
hold off



logDisplOct=log(displOct);

Rcta=polyfit(logDisplOct,hwyOct,1);
x1=log(1.5);
x2=log(6.5);
y1=polyval(Rcta,x1);
y2=polyval(Rcta,x2);

plot(logDisplOct,hwyOct,'o')
hold on
plot([x1 x2],[y1 y2],'r')
xlabel('log(Desplazamiento del motor [litros])')
ylabel('Rendimiento en autopista [millas/galón]')
hold off

rho=corr(hwyOct,displOct)
rhoS=corr(hwyOct,sqDisplOct)
rhoL=corr(hwyOct,logDisplOct)
