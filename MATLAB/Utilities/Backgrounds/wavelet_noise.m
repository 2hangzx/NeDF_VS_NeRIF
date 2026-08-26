function wavelet_noise_image = wavelet_noise(k_values, seed)
    rng(seed);
    % 设置k的范围
    % k_values = 3:11;
    % 设置图像大小
    image_size = 2^(max(k_values));
    
    % 初始化波纹噪声图像
    wavelet_noise_image = zeros(image_size);
    
    % 生成波纹噪声图像
    for k = k_values
        % 生成随机矩阵
        random_matrix = rand(2^k);
        
        % 将矩阵降采样至 2^(k-1) × 2^(k-1)
        downsampled_matrix = imresize(random_matrix, 0.5);
        
        % 将降采样后的矩阵插值至原始大小
        upsampled_matrix = imresize(downsampled_matrix, [2^k, 2^k]);
           
        % 计算波纹噪声图像的每一项
        wavelet_noise_term = random_matrix - upsampled_matrix;
    
        % 上采样回原始像素
        up_wavelet_noise_term = imresize(wavelet_noise_term, [image_size, image_size]);
        
        % 将每一项添加到波纹噪声图像中
        wavelet_noise_image = wavelet_noise_image + up_wavelet_noise_term;
    end
end

