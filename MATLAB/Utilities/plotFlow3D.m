function [surfacevalue,fh,cmap_index ] = plotFlow3D(flow, X, Y, Z, varargin)
% PLOTFLOW3D plots a 3D flow surface in the same value
%   PLOFLOW(flow) plots 3D flow looping thourgh X axis (first
%   dimension)
%   PLOTFLOW(flow,OPT,VAL,...) uses options and values for plotting. The
%   possible options in OPT are:
%
%   
%   'Colormap': Sets the colormap. Possible values for VAL are the names of
%               the stadard MATLAB colormaps, the names in the perceptually 
%               uniform colormaps tool or a custom colormap, being this last 
%               one a 3xN matrix. Default is GRAY
%   'Surfacevalue':     
%               a Nx1 matrix setting the value of the isosurface. The default 
%               computes the 5 equant values between the lower and upper 
%               percentile of data, in 1% and 99%.
%   'Facealpha':     
%               a Nx1 matrix setting the face alpha of the isosurface. The 
%               default computes the N equant values between 0.5 and 0.1.
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------
% This file is part of the BOSLAB Toolbox
% 
% Copyright (c) 2015, Beihang University
%                     All rights reserved.
%
% License:            Open Source under BSD. 
%                     See the full license at
%                     https:XXXXX
%
% Contact:            Weihu22@buaa.edu.cn
% Codes:              https:XXXXX
% Coded by:           Wei Hu
%--------------------------------------------------------------------------
%% Parse inputs

opts=     {'colormap','surfacevalue','facealpha','savegif','viewangle','figure_flag'};
defaults= [   1  ,      1,             1,         1,          1,        1];

% Check inputs
nVarargs = length(varargin);
if mod(nVarargs,2)
    error('BOSLAB:plotFlow3D:InvalidInput','Invalid number of inputs')
end

% check if option has been passed as input
for ii=1:2:nVarargs
    ind=find(ismember(opts,lower(varargin{ii})));
    if ~isempty(ind)
       defaults(ind)=0; 
    else
       error('BOSLAB:plotFlow3D:InvalidInput',['Optional parameter "' varargin{ii} '" does not exist' ]); 
    end
end

ninput=1;
for ii=1:length(opts)
    opt=opts{ii};
    default=defaults(ii);
    % if one option isnot default, then extranc value from input
    if default==0
        ind=double.empty(0,1);jj=1;
        while isempty(ind)
            ind=find(isequal(opt,lower(varargin{jj})));
            jj=jj+1;
        end
        if isempty(ind)
           error('BOSLAB:plotFlow3D:InvalidInput',['Optional parameter "' varargin{jj} '" does not exist' ]); 
        end
        val=varargin{jj};
    end
    
    switch opt
% % % % % %         % Colormap choice
        case 'colormap'
            if default
                cmap='gray';
            else
                
                if ~isnumeric(val)  
                    % check if it is from perceptually uniform colormaps.
                    if ismember(val,{'magma','viridis','plasma','inferno'})
                        cmap=eval([val,'()']);
                    else
                        cmap=val;
                    end                   
                else
                    % if it is a custom colormap
                    if size(val,2)~=3
                        error('BOSLAB:plotFlow3D:InvalidInput','Invalid size of colormap')
                    end
                    cmap=val;
                end
            end
% % % % % % %         % do you want to save result as gif?
        case 'savegif'
            if default
                savegif=0;
            else
               savegif=1;
               if ~ischar(val)
                   error('BOSLAB:plotFlows:InvalidInput','filename is not character')
               end
               filename=val;
            end
% % % % % % %         % do you want to save result as gif?
        case 'viewangle'
            if default
                viewangle=[-37.5, 30];
            else
               viewangle=val;
            end
% % % % % %         % IsoSurface value
        case 'surfacevalue'
            if default                
                    climits=prctile(double(flow(:)),[1 99]);
                    Surfacevalue =linspace(climits(1), climits(2), 5);
            else              
                    climits=[min(val), max(val)];
                    Surfacevalue = val;
            end
% % % % % %         % face alpha
        case 'facealpha'
            if default 
                    faceN = length(Surfacevalue);
                    Facealpha= linspace(0.5, 0.1, faceN);
            else              
                    Facealpha = val;
            end
% % % % % %         % face alpha
        case 'figure_flag'
            if default 
                    figure_flag = 1;
            else              
                    figure_flag = val;
            end            
        otherwise
          error('BOSLAB:plotFlow3D:InvalidInput',['Invalid input name:', num2str(opt),'\n No such option in plotFlow()']);
    end
end
%% Do the plotting!
if figure_flag
    fh=figure();
else
    fh = gcf;
end
hold on
surface_validindex = [];
for j =  length(Surfacevalue): -1: 1
    s = isosurface(Z, X, Y, flow, Surfacevalue(j));
    if ~isempty(s.faces)
        [faces,verts,colors] = isosurface(Z, X, Y, flow, Surfacevalue(j),Z);
        colors = zeros(size(colors)) + Surfacevalue(j);
        patch('Vertices',verts,'Faces',faces,'FaceVertexCData',colors,...
            'FaceColor','interp','EdgeColor','none', 'FaceAlpha',Facealpha(j));
        surface_validindex = [surface_validindex j];
    end
end

xlabel('\itz');
ylabel('\itx');
zlabel('\ity');
xlim([min(Z(:)), max(Z(:))]);
ylim([min(X(:)), max(X(:))]);
zlim([min(Y(:)), max(Y(:))]);
set(gca,'ZDir','reverse');
set(gca,'YDir','reverse');
axis equal

currentColormap = colormap(cmap); 
% calculate the color index of isoface in the current Colormap
index = round(( sort(Surfacevalue(surface_validindex))- min(Surfacevalue)) / (max(Surfacevalue) - min(Surfacevalue)) * (size(currentColormap, 1) - 1)) + 1;
cmap_index = currentColormap(index, :);
colormap(cmap_index); 
cb = colorbar; 
set(cb, 'ticks',sort(Surfacevalue(surface_validindex)));
view(viewangle);
surfacevalue = Surfacevalue(surface_validindex);

if savegif
  t = viewangle(1):10:360+viewangle(1);      
  for fi = 1:length(t)
    view(t(fi),viewangle(2));
    CurrFrame = getframe(fh);   
    im = frame2im(CurrFrame);  
    [A,map] = rgb2ind(im,256);  
	if fi == 1
		imwrite(A,map,filename,'gif','LoopCount',Inf,'DelayTime',0.1);  % DelayTime表示写入的时间间隔
	else
		imwrite(A,map,filename,'gif','WriteMode','append','DelayTime',0.1);
    end
  end
end
end
