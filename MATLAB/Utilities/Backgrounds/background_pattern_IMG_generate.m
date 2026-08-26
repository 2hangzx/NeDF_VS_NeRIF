function [IM_Interest,density_ob] = background_pattern_IMG_generate(pattern_type, imgsize, varargin)
    %% poisson dot
    if strcmp(pattern_type, 'poisson_dot')
        density = 0.4;
        radim =1;%2
        seed = 1234;%1985
        rng(seed);
        IM_Interest=makedotpatternIMG(density,radim, imgsize);
        density_ob = sum(1-IM_Interest(:))/imgsize(1)/imgsize(2);
    end
     %% DLA
    if strcmp(pattern_type, 'DLA')
        density = 0.4;
        IM = diffusion_limited_aggregation(imgsize,density);
        IM_gray = rgb2gray(IM);
        if any(strcmp(varargin, 'gray_threshold'))
            id = find(strcmp(varargin, 'gray_threshold'));
            threshold = varargin{id+1}; 
        else
            threshold = 0.1;
        end
        IM_Interest = double(~imbinarize(IM_gray, threshold));
        density_ob = sum(1-IM_Interest(:))/imgsize(1)/imgsize(2);
    end
    %% gosper curve
    if strcmp(pattern_type, 'gosper')  

        line_thickness = 1; 
        matrixCo=2.6;%2.7
        IM = gosper_curve_IMG(imgsize, matrixCo, line_thickness);

        f1=figure(1); clf;
        imshow(IM);
        axis equal
        pos = [0 0 imgsize(1)-1 imgsize(2)-1];
        h =  drawrectangle('Position', pos, 'LineWidth',1,'Color','r','Label','ROI');
        pos = round(customWait(h));
        close(f1);
        IM_Interest = IM(pos(2):pos(2)+pos(4),pos(1):pos(1)+pos(3));        
        density_ob = sum(1-IM_Interest(:))/imgsize(1)/imgsize(2);
    end
     %% hilbert curve
    if strcmp(pattern_type, 'hilbert')  

        line_thickness = 1; 
        matrixCo=4;%2.7
        IM = hilbert_curve_IMG(imgsize, matrixCo, line_thickness);

        f1=figure(1); clf;
        imshow(IM);
        axis equal
        pos = [1 1 imgsize(1)-1 imgsize(2)-1];
        h =  drawrectangle('Position', pos, 'LineWidth',1,'Color','r','Label','ROI');
        pos = round(customWait(h));
        close(f1);
        IM_Interest = IM(pos(2):pos(2)+pos(4),pos(1):pos(1)+pos(3));    
        density_ob = sum(1-IM_Interest(:))/imgsize(1)/imgsize(2);
    end
    %% sinusoidal
    if strcmp(pattern_type, 'sinusoidal')
        [x,y] = meshgrid(0:imgsize(1)-1, 0:imgsize(2)-1);
        f0 = 1/5;%1/17
        IM = 1/2 + ( cos(2*pi*f0*x) + cos(2*pi*f0*y) )/4;% + ( cos(2*pi*f1*x) + cos(2*pi*f1*y) )/8 ;
        IM_Interest = double(imbinarize(IM,0.4));
        density_ob = sum(1-IM_Interest(:))/imgsize(1)/imgsize(2);        
    end
    %% wavelet
    if strcmp(pattern_type, 'wavelet_noise')
        MaxLevel = max(nextpow2(imgsize(1)), nextpow2(imgsize(2)));      
        MinLevel = 3;  
        seed = 1985;
        k_values = MinLevel:MaxLevel;
        im = mat2gray( wavelet_noise(k_values, seed) );
        f1=figure(1); clf;
        imshow(im); 
        axis equal
    
        if 2^MaxLevel == imgsize(1) && 2^MaxLevel == imgsize(2)
            im_ROI = im;
        else
            pos = [0 0 imgsize(1)-1 imgsize(2)-1];
            h =  drawrectangle('Position', pos, 'LineWidth',1,'Color','r','Label','ROI');
            pos = round(customWait(h));
            IM_ROI = im(pos(2):pos(2)+pos(4),pos(1):pos(1)+pos(3));
        end
        close(f1);
        IM_Interest = imadjust((IM_ROI));
        density_ob = sum(1-IM_Interest(:))/imgsize(1)/imgsize(2); 
    end
end