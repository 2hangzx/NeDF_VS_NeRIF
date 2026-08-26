function plot_slices(sigmas_path, varargin)
% PLOT_SLICES  从 sigmas0.mat 绘制 3D 折射率场的切片图
%
% 用法:
%   plot_slices('path/to/sigmas0.mat')
%       默认：输出 X / Y / Z 三个方向的中截面各一张
%
%   plot_slices('path/to/sigmas0.mat', 'x', 70)
%       X 方向第 70 层（单张）
%
%   plot_slices('path/to/sigmas0.mat', 'y', [50, 100, 150])
%       Y 方向三个切片（排成一行 subplot）
%
%   plot_slices('path/to/sigmas0.mat', 'z', 1:20:140)
%       Z 方向每隔 20 层取一张
%
%   plot_slices('path/to/sigmas0.mat', 'x', 70, 'clim', [-0.5, 2.5])
%       自定义 colorbar 范围
%
% 输入:
%   sigmas_path  - sigmas0.mat 文件路径（字符串）
%   'x'/'y'/'z' - 切片方向（可选，默认三个方向都出）
%   索引         - 整数或整数数组（可选，默认取中截面）
%   'clim'       - [cmin, cmax] 颜色范围（可选键值对）
%   'outdir'     - 输出目录（可选，默认与 sigmas0.mat 同目录）
%
% 输出:
%   切片 PNG 文件，命名如 slice_x070.png, slice_y_mid.png 等

    % ===== 解析参数 =====
    p = inputParser;
    p.addRequired('sigmas_path', @(x) ischar(x) || isstring(x));
    p.addOptional('direction',  '', @(x) ischar(x) || isstring(x));
    p.addOptional('indices',    [], @(x) isnumeric(x));
    p.addParameter('clim',    [], @(x) isnumeric(x) && numel(x) == 2);
    p.addParameter('outdir',  '', @(x) ischar(x) || isstring(x));
    p.parse(sigmas_path, varargin{:});

    sigmas_path = p.Results.sigmas_path;
    direction   = lower(char(p.Results.direction));
    indices     = p.Results.indices;
    clim_user   = p.Results.clim;
    outdir      = char(p.Results.outdir);

    % ===== 加载数据 =====
    if ~isfile(sigmas_path)
        error('File not found: %s', sigmas_path);
    end
    data = load(sigmas_path);
    if ~isfield(data, 'sigmas0')
        error('Variable ''sigmas0'' not found in %s. Available: %s', ...
              sigmas_path, strjoin(fieldnames(data), ', '));
    end
    sigmas = data.sigmas0;  % [nx, ny, nz]

    [nx, ny, nz] = size(sigmas);
    fprintf('[plot_slices] Loaded sigmas0: %d x %d x %d\n', nx, ny, nz);
    fprintf('[plot_slices] Value range: [%.4f, %.4f]\n', min(sigmas(:)), max(sigmas(:)));

    % ===== 默认输出目录 =====
    if isempty(outdir)
        [filepath, ~] = fileparts(sigmas_path);
        if isempty(filepath)
            filepath = '.';
        end
        outdir = fullfile(filepath, 'slices');
    end
    if ~exist(outdir, 'dir')
        mkdir(outdir);
    end

    % ===== 默认颜色范围 =====
    if isempty(clim_user)
        clim_user = [min(sigmas(:)), max(sigmas(:))];
    end

    % ===== 确定切片列表 =====
    if isempty(direction)
        % 默认：三方向中截面
        slice_list = {
            'x', round(nx/2);
            'y', round(ny/2);
            'z', round(nz/2);
        };
    else
        if ~ismember(direction, {'x', 'y', 'z'})
            error('Direction must be ''x'', ''y'', or ''z'', got ''%s''', direction);
        end
        if isempty(indices)
            % 未指定索引 → 取中截面
            switch direction
                case 'x', indices = round(nx/2);
                case 'y', indices = round(ny/2);
                case 'z', indices = round(nz/2);
            end
        end
        indices = unique(max(1, min(indices, get_dim(direction, nx, ny, nz))));
        slice_list = [repmat({direction}, numel(indices), 1), num2cell(indices(:))];
    end

    % ===== 逐张绘制 =====
    for s = 1:size(slice_list, 1)
        dir_i  = slice_list{s, 1};
        idx_i  = slice_list{s, 2};

        % 提取切片
        switch dir_i
            case 'x'
                slice_data = squeeze(sigmas(idx_i, :, :))';  % [nz, ny] → [ny, nz]
                xlabel_str = 'Y'; ylabel_str = 'Z';
                sz1 = ny; sz2 = nz;
            case 'y'
                slice_data = squeeze(sigmas(:, idx_i, :))';  % [nx, nz] → [nz, nx]
                xlabel_str = 'X'; ylabel_str = 'Z';
                sz1 = nx; sz2 = nz;
            case 'z'
                slice_data = squeeze(sigmas(:, :, idx_i))';  % [nx, ny] → [ny, nx]
                xlabel_str = 'X'; ylabel_str = 'Y';
                sz1 = nx; sz2 = ny;
        end

        % 文件名
        if numel(idx_i) == 1 && isscalar(idx_i)
            fname = sprintf('slice_%s%03d.png', dir_i, idx_i);
        else
            fname = sprintf('slice_%s_mid.png', dir_i);
        end
        out_path = fullfile(outdir, fname);

        % 绘图
        figure('Visible', 'off', 'Position', [100, 100, 800, 600]);
        imagesc(slice_data, clim_user);
        axis equal tight;
        colormap(jet(256));
        cb = colorbar;
        ylabel(cb, 'Refractive index \Delta n', 'FontSize', 11);
        xlabel(xlabel_str, 'FontSize', 12);
        ylabel(ylabel_str, 'FontSize', 12);
        title(sprintf('Slice %s = %d / %d', upper(dir_i), idx_i, ...
              get_dim(dir_i, nx, ny, nz)), 'FontSize', 13);

        % 保存
        exportgraphics(gcf, out_path, 'Resolution', 150);
        close(gcf);
        fprintf('[plot_slices] Saved: %s\n', out_path);
    end

    fprintf('[plot_slices] Done. %d slice(s) saved to %s\n', size(slice_list, 1), outdir);
end

% ===== 辅助函数 =====
function d = get_dim(dir_char, nx, ny, nz)
    switch dir_char
        case 'x', d = nx;
        case 'y', d = ny;
        case 'z', d = nz;
        otherwise, error('Invalid direction: %s', dir_char);
    end
end
