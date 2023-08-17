function distance = canberra(rd,pd)
%CANBERRA	  Calculate the average Canberra between 
%               the predicted and the real label distribution.
%
%	Description
%   DISTANCE = Canberra(RD, PD) calculate the average Canberra
%   between the predicted and the real label distribution.
%
%   Inputs,
%       RD: real label distribution
%       PD: predicted label distribution
%
%   Outputs,
%       DISTANCE: Canberra
%
% rd = rd + 1e-7;
% pd = pd + 1e-7;
temp=abs(rd-pd);
temp2=abs(pd)+abs(rd);
temp2 = temp2 + 1e-7; %
temp=temp./temp2;
temp=sum(temp,2);
distance=temp;
% distance=mean(temp);
end

