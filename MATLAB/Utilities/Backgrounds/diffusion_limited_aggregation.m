function IM = diffusion_limited_aggregation(image_size, sol_density)
    DIM = image_size;   
    seed = 1985;
    t_max = 180;
    dt = 18;

    % 
    % SOLID = zeros(DIM)';
    % SOLID(:,1) = 1;
    % SOLID(:,end) = 1;
    % SOLID(1,:) = 1;
    % SOLID(end,:) = 1;
    % dla_simulation(SOLID, 0.15, 1985, 750, 10, 'dla-box');
    % 
    SOLID = zeros(DIM)';
    SOLID(ceil(DIM(2)/2),ceil(DIM(1)/2)) = 1;
    dla_simulation(SOLID, sol_density, seed, t_max, dt, 'dla-center');
    % 
    % SOLID = zeros(DIM)';
    % SOLID(ceil(0.25*DIM(2)),ceil(0.25*DIM(1))) = 1;
    % SOLID(ceil(0.75*DIM(2)),ceil(0.25*DIM(1))) = 1;
    % SOLID(ceil(0.25*DIM(2)),ceil(0.75*DIM(1))) = 1;
    % SOLID(ceil(0.75*DIM(2)),ceil(0.75*DIM(1))) = 1;
    % dla_simulation(SOLID, 0.15, 1985, 750, 10, 'dla-four');
    % 
    % SOLID = zeros(DIM)';
    % SOLID(ceil(DIM(2)/2),:) = 1;
    % dla_simulation(SOLID, 0.15, 1985, 750, 10, 'dla-line');
    
    % dla_animation('dla-box',60);
    IM = dla_animation('dla-center',60);
end
% dla_animation('dla-four',60);
% dla_animation('dla-line',60);


% SOLID = zeros(DIM)';
% SOLID(end,ceil(DIM(1)/2)) = 1;
% dla_simulation_nopbc(SOLID, 0.15, 1985, 2000, 10, 'dla-bottom-nopbc');
% dla_animation('dla-bottom-nopbc',60);

% SOLID = zeros(DIM)';
% SOLID(ceil(DIM(2)/2),ceil(DIM(1)/2)) = 1;
% dla_simulation_nopbc(SOLID, 0.15, 1985, 1500, 10, 'dla-center-nopbc');
% dla_animation('dla-center-nopbc',60);

% SOLID = zeros(DIM)';
% SOLID(ceil(0.25*DIM(2)),ceil(0.25*DIM(1))) = 1;
% SOLID(ceil(0.75*DIM(2)),ceil(0.25*DIM(1))) = 1;
% SOLID(ceil(0.25*DIM(2)),ceil(0.75*DIM(1))) = 1;
% SOLID(ceil(0.75*DIM(2)),ceil(0.75*DIM(1))) = 1;
% dla_simulation_nopbc(SOLID, 0.15, 1985, 750, 10, 'dla-four-nopbc');
% dla_animation('dla-four-nopbc',60);

% SOLID = zeros(DIM)';
% SOLID(ceil(DIM(2)/2),:) = 1;
% dla_simulation_nopbc(SOLID, 0.15, 1985, 750, 10, 'dla-line-nopbc');
% dla_animation('dla-line-nopbc',60);
% 
% SOLID = zeros(DIM)';
% SOLID(end,:) = 1;
% dla_simulation_nopbc(SOLID, 0.15, 1985, 1500, 10, 'dla-line_bottom-nopbc');
% dla_animation('dla-line_bottom-nopbc',60);

% SOLID = zeros(DIM)';
% SOLID(ceil(DIM(2)/2),1) = 1;
% dla_simulation_nopbc(SOLID, 0.15, 1985, 3000, 10, 'dla-side-nopbc');
% dla_animation('dla-side-nopbc',60);