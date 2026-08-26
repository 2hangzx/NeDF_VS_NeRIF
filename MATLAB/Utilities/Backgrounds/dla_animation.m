function IM = dla_animation(FILENAME,framerate)
video_flag = false;
if video_flag
    % Initializing video objects:
    vid = VideoWriter(FILENAME,'Uncompressed AVI');
    vid.FrameRate = framerate;
    open(vid);
end
% Initializing mask:
mask_size = 51;
r = (mask_size + 1)/2;
MASK = zeros(mask_size);
for x = 1:mask_size
    for y = 1:mask_size
        MASK(x,y) = single((x-r)^2 + (y-r)^2 < r^2);
    end
end
MASK = MASK/sum(MASK(:));

% Initializing frame:
load(FILENAME,'SOLID','SOLUTION');
DIM = size(SOLID);
l_l = 1:r;
l_r = (DIM(2)+1-r):DIM(2);
l_u = 1:r;
l_d = (DIM(1)+1-r):DIM(1);

SOLID_s = double(SOLID(:,:,1));
SOLID_frame = hsv2rgb(cat(3, SOLID_s./DIM(3), SOLID_s, SOLID_s));

SOLUTION_s = double(SOLUTION(:,:,1));
SOLUTION_pbc = [SOLUTION_s(l_u,l_l), SOLUTION_s(l_u,:), SOLUTION_s(l_u,l_r); ...
                SOLUTION_s(:,l_l),   SOLUTION_s,        SOLUTION_s(:,l_r);   ...
                SOLUTION_s(l_d,l_l), SOLUTION_s(l_d,:), SOLUTION_s(l_d,l_r)];
SOLUTION_pbc = conv2(SOLUTION_pbc,MASK,'same');
SOLUTION_s = SOLUTION_pbc(r+1:(end-r),r+1:(end-r));
SOLUTION_s(SOLID(:,:,1)) = 0;
SOLUTION_frame = cat(3, SOLUTION_s, SOLUTION_s, SOLUTION_s);

FRAME = im2frame(SOLUTION_frame + SOLID_frame);
if video_flag
    writeVideo(vid,FRAME);
end
for t = 2:DIM(3)
    SOLID_s = SOLID_s + double(SOLID(:,:,t)-SOLID(:,:,t-1))*t/DIM(3);
    SOLID_bg = double(SOLID_s > 0);
    SOLID_frame = hsv2rgb(cat(3, SOLID_s, SOLID_bg, SOLID_bg));
    
    SOLUTION_s = double(SOLUTION(:,:,t));
    SOLUTION_pbc = [SOLUTION_s(l_u,l_l), SOLUTION_s(l_u,:), SOLUTION_s(l_u,l_r); ...
                    SOLUTION_s(:,l_l),   SOLUTION_s,        SOLUTION_s(:,l_r);   ...
                    SOLUTION_s(l_d,l_l), SOLUTION_s(l_d,:), SOLUTION_s(l_d,l_r)];
    SOLUTION_pbc = conv2(SOLUTION_pbc,MASK,'same');
    SOLUTION_s = SOLUTION_pbc(r+1:(end-r),r+1:(end-r));
    SOLUTION_s(SOLID(:,:,t)) = 0;
    SOLUTION_frame = cat(3, SOLUTION_s, SOLUTION_s, SOLUTION_s);
    
    FRAME = im2frame(SOLUTION_frame + SOLID_frame);
    if video_flag
        writeVideo(vid,FRAME);
    end
    t/DIM(3);
end
if video_flag
    close(vid);
end
IM = FRAME.cdata;
