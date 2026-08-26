function IM = hilbert_curve_IMG(imgsize, matrixCo, line_thickness)
    imgsize = [max(imgsize),max(imgsize)];
    order = 8;%     
    [x,y] = hilbert_curve(order);
    x = (x - min(x)) / (max(x) - min(x));
    y = (y - min(y)) / (max(y) - min(y));
    matrix_size = round(imgsize*matrixCo);
    xp= round(x*(matrix_size(1)-1))+1;
    yp= round(y*(matrix_size(2)-1))+1;  

    IM = zeros(matrix_size)+1;
    for i= 1:1: length(xp)-1
        IM(yp(i),xp(i))=0;
        IM(yp(i+1),xp(i+1))=0;
        % 使用 Bresenham 算法绘制线
      
        % 定义点 a 和 b
        a = [xp(i), yp(i)];  % 起点 (x1, y1)
        b = [xp(i+1), yp(i+1)];  % 终点 (x2, y2)
       
        % 计算线段的长度
        num_points = max(abs(b(1) - a(1)), abs(b(2) - a(2))) + 1;
        
        % 使用 linspace 生成线段上的 x 和 y 坐标
        x_coords = round(linspace(a(1), b(1), num_points));
        y_coords = round(linspace(a(2), b(2), num_points));
        
        % 设置线条厚度：对每个点，增加厚度范围内的周围像素
        for k = 1:num_points
            % 获取当前点的 x, y 坐标
            x = x_coords(k);
            y = y_coords(k);
            
            % 在厚度范围内设置邻近的像素为黑色
            for dx = -line_thickness:line_thickness
                for dy = -line_thickness:line_thickness
                    % 确保坐标不超出矩阵范围
                    if (x + dx > 0 && x + dx <= matrix_size(2) && y + dy > 0 && y + dy <= matrix_size(1))
                        IM(y + dy, x + dx) = 0;  % 设置为黑色
                    end
                end
            end
        end
    end

end