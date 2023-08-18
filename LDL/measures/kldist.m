function distance = kldist(rd, pd)
%KLDIST	  Calculate the average Kullback-Leibler divergence between the predicted label
%         distribution and the real label distribution.
%
%	Description
%   DISTANCE = KLDIST(RD, PD) calculate the average Kullback-Leibler divergence 
%   between the predicted label distribution and the real label distribution.
%
%   Inputs,
%       RD: real label distribution
%       PD: predicted label distribution
%
%   Outputs,
%       DISTANCE: Kullback-Leibler divergence
%	
eps = 1e-7;
temp=rd.*(log((rd + eps)./(pd + eps)));
distance = sum(temp);
% distance=mean(sum(temp,2));
end
