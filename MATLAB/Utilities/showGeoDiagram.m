function  showGeoDiagram()
%SHOWGEODIAGRAM Shows an image describing the Geometry of BOST
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
%%
figure('Name','Diagram of BOST Geometry');
title('Diagram of BOST Geometry');
geoimg=imread('Diagram of the geometry of BOST.jpg');
imshow(geoimg);

h = xlabel(''); 
pos = get(h,'Position'); 
delete(h)
h = title(char('Geometry definition for BOST,  DOI: XXXXX',...
    'Current BOST is more flexible than what is shown in the figure.'));
set(h,'Position',pos);
set(gca, 'XAxisLocation','top');
set(gcf, 'Color','white');


%%
figure('Name','Transformation between coordinate systems');
title('Transformation between coordinate systems');
geoimg=imread('Transformation between coordinate systems.jpg');
imshow(geoimg);

h = xlabel(''); 
pos = get(h,'Position'); 
delete(h)
h = title(char('Transformation between coordinate systems'));
set(h,'Position',pos);
set(gca, 'XAxisLocation','top');
set(gcf, 'Color','white');
%%
figure('Name','Default gemetry of simu BOST');
title('Default gemetry of simu BOST');
geoimg=imread('default gemetry.jpg');
imshow(geoimg);

h = xlabel(''); 
pos = get(h,'Position'); 
delete(h)
h = title(char('Default gemetry of simu BOST'));
set(h,'Position',pos);
set(gca, 'XAxisLocation','top');
set(gcf, 'Color','white');