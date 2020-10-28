df=importdata('HW3Data.csv');

 x1=df(:,1);
x2=df(:,2);
y=df(:,3);
sum1=0;
sum2=0;
 


for i=1:1200
  if y(i,1)==1
      sum1=sum1+1;
       plot(x1(i,1),x2(i,1),'ko','Color','r','MarkerSize',4)
       hold on
          class1(sum1,1)=(x1(i,1));
          class1(sum1,2)=(x2(i,1));
     
  else 
      sum2=sum2+1;
      plot(x1(i,1),x2(i,1),'ko','Color','b','MarkerSize',4)
       hold on
      
        class2(sum2,1)=(x1(i,1));
        class2(sum2,2)=(x2(i,1));
          
  end
 
 
end
 
error_func=ones((sum2+sum1),1);
error=sum(error_func);
N=sum1+sum2;

X0=ones((sum2+sum1),1);  %for bias input
X1=[class1(:,1);class2(:,1)];
X2=[class1(:,2);class2(:,2)];
X_general_unnormalize=[X0 X1 X2];
X_general=[ X0 mat2gray(X1) mat2gray(X2)]';
Ygen=[ones(size(class1(:,1))) ; zeros(size(class2(:,1)))]';

%backpropagation algorithm initial values

H=3; %Hidden neuron number
K=1; %Output neuron number
I=3; %Input neuron number : Bias,X1,X2

W=randi([5,10],H,I)./100; %initial w matrix
V=randi([5,10],K,H)./100; %initial v matrix

d_W=zeros(H,I); 
d_V=zeros(K,H); 

%sigmoid
z=zeros(H,N);

y=zeros(K,N);

epsilon=0.01;
learning_rate=15;
% alfa=0.5;
iteration=0;
maxiteration=10000;
%backpropagation algorithm initial values



%backpropagation algorithm implementation
while error>epsilon  && (maxiteration>iteration)
   iteration= iteration+1;
    d_W=zeros(H,I);
    d_V=zeros(K,H);

    for i=1:1:N
        %z calculation
        for h=1:1:H
            z(h,i)=1/(1+exp(-W(h,:)*X_general(:,i)));
        end
        %y calculation
        for k=1:1:K
            y(k,i)=1/(1+exp(-(V(k,:)*z(:,i))));
        end
  
%d_v
   for k=1:1:K
       for h=1:1:H
            d_V(k,h)=d_V(k,h)+learning_rate*(Ygen(k,i)-y(k,i))*z(h,i);
        
       end
       
   end
for h=1:1:H
    for in=1:1:I
        d_W(h,in)=d_W(h,in)+learning_rate*(Ygen(:,i)-y(:,i))'*V(:,h)*z(h,i)*(1-z(h,i))*X_general(in,i);
%         d_W(h,in)=d_W(h,in)-(alfa)*(y(k,i)-(Ygen(:,i)-y(:,i))'*V(:,h)*z(h,i)*(1-z(h,i))*X_general(in,i);
    end
end

error_func(i,1)=sum(1/2*(Ygen(:,i)-y(:,i)).^2);
    end
error=sum(error_func)/N;
V=V+d_V/N;
W=W+d_W/N;
end

figure
Y_plot=zeros(1,2000);
Y_plot(1:1003)=1;
Y_plot(1004:2000)=0;

plot(Y_plot);

hold on

