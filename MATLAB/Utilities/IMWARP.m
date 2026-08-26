function [iref, idis] = IMWARP(iref0, projectionsU, projectionsV, geo, interp_method)
    clear idis iref
    for i = 1: 1: geo.numCam
        if size(iref0,3)==1
            iref(:,:,i) = im2double(iref0);
            idis(:,:,i) = interp_imwarp(iref0, projectionsU(:,:,i), projectionsV(:,:,i), interp_method);
        else
            iref(:,:,i) = im2double(iref0(:,:,i));
            idis(:,:,i) = interp_imwarp(iref0(:,:,i), projectionsU(:,:,i), projectionsV(:,:,i), interp_method);
        end
    end
end