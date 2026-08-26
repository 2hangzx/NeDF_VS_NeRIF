function dla_simulation(SOLID_0, sol_density, seed, t_max, dt, FILENAME)
% dla_simulation(SOLID_0, sol_density, seed, t_max, dt, FILENAME)
% SOLID_0
% sol_density
% seed
% t_max
% dt
% FILENAME


% Constants:
DIM = size(SOLID_0);
i = 1:2:DIM(1)-1;
j = 1:2:DIM(2)-1;
ip = i+1;
jp = j+1;
length_i = length(i);
length_j = length(j);

% Initializing matrixes:
rng(seed);
SOLID = zeros([DIM, t_max],'logical');
SOLID(:,:,1) = SOLID_0;
SOLUTION = SOLID;
SOLUTION(:,:,1) = (rand(DIM) < sol_density);
SOLUTION(SOLID) = 0;
BUFFER_solution_2 = SOLUTION(:,:,1);

% Main simulation:
for t = 2:t_max
    BUFFER_solution = SOLUTION(:,:,t-1);
    BUFFER_solid = SOLID(:,:,t-1);
    for t2 = 2:dt
        % Margulous Neiboourhood for odd steps:
        ROT_clock = logical(randi(2,length_i,length_j)-1);
        ROT_antic = ~ROT_clock;    
        BUFFER_solution_2(i,j)   = (ROT_clock&BUFFER_solution(ip,j)) | ...
                                   (ROT_antic&BUFFER_solution(i,jp));
        BUFFER_solution_2(i,jp)  = (ROT_clock&BUFFER_solution(i,j)) | ...
                                   (ROT_antic&BUFFER_solution(ip,jp));
        BUFFER_solution_2(ip,j)  = (ROT_clock&BUFFER_solution(ip,jp)) | ...
                                   (ROT_antic&BUFFER_solution(i,j));
        BUFFER_solution_2(ip,jp) = (ROT_clock&BUFFER_solution(i,jp)) | ...
                                   (ROT_antic&BUFFER_solution(ip,j));
        BUFFER_solution = BUFFER_solution_2;
        SOLID_new = BUFFER_solid;
        while any(SOLID_new(:))
            NEIGHBOR = circshift(BUFFER_solid,[ 0, 1]) | ...
                       circshift(BUFFER_solid,[ 0,-1]) | ...
                       circshift(BUFFER_solid,[ 1, 0]) | ...
                       circshift(BUFFER_solid,[ -1, 0]);
            SOLID_new = NEIGHBOR&BUFFER_solution;
            BUFFER_solid = BUFFER_solid | SOLID_new;
            BUFFER_solution(BUFFER_solid) = 0;
        end
        
        % Margulous Neiboourhood for even steps:
        ROT_clock = (rand(length_i,length_j) > 0.5);
        ROT_antic = ~ROT_clock;
        BUFFER_solution = circshift(BUFFER_solution,[-1,-1]);
        BUFFER_solution_2(i,j)   = (ROT_clock&BUFFER_solution(ip,j)) | ...
                                   (ROT_antic&BUFFER_solution(i,jp));
        BUFFER_solution_2(i,jp)  = (ROT_clock&BUFFER_solution(i,j)) | ...
                                   (ROT_antic&BUFFER_solution(ip,jp));
        BUFFER_solution_2(ip,j)  = (ROT_clock&BUFFER_solution(ip,jp)) | ...
                                   (ROT_antic&BUFFER_solution(i,j));
        BUFFER_solution_2(ip,jp) = (ROT_clock&BUFFER_solution(i,jp)) | ...
                                   (ROT_antic&BUFFER_solution(ip,j));
        BUFFER_solution = circshift(BUFFER_solution_2,[1,1]);
        SOLID_new = BUFFER_solid;
        while any(SOLID_new(:))
            NEIGHBOR = circshift(BUFFER_solid,[ 0, 1]) | ...
                       circshift(BUFFER_solid,[ 0,-1]) | ...
                       circshift(BUFFER_solid,[ 1, 0]) | ...
                       circshift(BUFFER_solid,[ -1, 0]);
            SOLID_new = NEIGHBOR&BUFFER_solution;
            BUFFER_solid = BUFFER_solid | SOLID_new;
            BUFFER_solution(BUFFER_solid) = 0;
        end
    end
    SOLUTION(:,:,t) = BUFFER_solution;
    SOLID(:,:,t) = BUFFER_solid;
end
save(FILENAME,'SOLUTION','SOLID');