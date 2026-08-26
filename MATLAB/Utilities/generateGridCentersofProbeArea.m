function [X, Y, Z] = generateGridCentersofProbeArea(geo)
% Given the coordinates of the large cuboid center, side lengths, and number of grids
x_c = geo.Opr(1); % x-coordinate of the cuboid center
y_c = geo.Opr(2); % y-coordinate of the cuboid center
z_c = geo.Opr(3); % z-coordinate of the cuboid center
Lx = geo.sVoxel(1); % Side length in the x-direction
Ly = geo.sVoxel(2);  % Side length in the y-direction
Lz = geo.sVoxel(3);  % Side length in the z-direction
nx = geo.nVoxel(1);  % Number of grids in the x-direction
ny = geo.nVoxel(2);  % Number of grids in the y-direction
nz = geo.nVoxel(3);  % Number of grids in the z-direction

% Calculate the size of each small cuboid
dx = Lx / nx;
dy = Ly / ny;
dz = Lz / nz;

% Calculate the center coordinates in each direction
x_range = linspace(-Lx/2 + dx/2, Lx/2 - dx/2, nx);
y_range = linspace(-Ly/2 + dy/2, Ly/2 - dy/2, ny);
z_range = linspace(-Lz/2 + dz/2, Lz/2 - dz/2, nz);

% Generate a 3D grid using ndgrid
% [Z, X, Y] = ndgrid(z_range, x_range, y_range);
[X, Y, Z] = ndgrid(x_range, y_range, z_range);
% Shift the grid coordinates to the center of the large cuboid
X = X + x_c;
Y = Y + y_c;
Z = Z + z_c;

% Combine the coordinates into the center coordinates of nx*ny*nz small cuboids
X = reshape(X, nx, ny, nz);
Y = reshape(Y, nx, ny, nz);
Z = reshape(Z, nx, ny, nz);
end