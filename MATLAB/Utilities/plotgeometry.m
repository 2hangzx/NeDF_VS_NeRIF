function h=plotgeometry(geo)
%PLOTGEOMETRY(GEO) plots a 3D version of the BOST geometry with the
% given geomerty GEO. 
% 
% h=PLOTGEOMETRY(...) will return the figure handle 

%%  Figure stuff
h=figure('Name','Background Oriented Schlieren Tomography geometry');
hold on
title('Current BOST geometry, in scale')
xlabel('Z');
ylabel('X');
zlabel('Y');
set(gcf, 'color', [1 1 1])
%% CUBE/Probe area
drawCube(geo.Opr,geo.sVoxel,'k',0.05);
%% CUBE/Background
for i = 1: 1: geo.numCam 
    PbkCorner = geo.PbkCorner(:, 4*(i-1)+1: 4*i);
    x = [PbkCorner(1,1) PbkCorner(1,2) PbkCorner(1,4) PbkCorner(1,3) PbkCorner(1,1)];
    y = [PbkCorner(2,1) PbkCorner(2,2) PbkCorner(2,4) PbkCorner(2,3) PbkCorner(2,1)];
    z = [PbkCorner(3,1) PbkCorner(3,2) PbkCorner(3,4) PbkCorner(3,3) PbkCorner(3,1)];
    patch(z, x, y, 'blue', 'FaceAlpha', 0.5);
    text(mean(z),mean(x),mean(y),num2str(i),'FontSize',12);
end
%% CUBE/camera
% optical center
for i = 1: 1: geo.numCam 
    x = geo.Ocr(1,i);
    y = geo.Ocr(2,i);
    z = geo.Ocr(3,i);
    plot3(z,x,y,'.','MarkerSize',20);
    text(z,x,y,num2str(i),'FontSize',12);
end
%%
axis equal;
set(gca,'YDir','reverse'); 
view(158,30);
end

function drawCube( origin, size,color,alpha)

x=([0 1 1 0 0 0;1 1 0 0 1 1;1 1 0 0 1 1;0 1 1 0 0 0]-0.5)*size(1)+origin(1);
y=([0 0 1 1 0 0;0 1 1 0 0 0;0 1 1 0 1 1;0 0 1 1 1 1]-0.5)*size(2)+origin(2);
z=([0 0 0 0 0 1;0 0 0 0 0 1;1 1 1 1 0 1;1 1 1 1 0 1]-0.5)*size(3)+origin(3);
for i=1:6
    h=patch(z(:,i),x(:,i),y(:,i),color);
    set(h,'facealpha',alpha)
end

end