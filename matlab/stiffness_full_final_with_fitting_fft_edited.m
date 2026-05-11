function stiffness_full_final_with_fitting_fft_edited(total_frame, variable_averaging)
    % 若未传入参数，使用默认值（便于单独调试）
    if nargin < 2
        total_frame = 40;
        variable_averaging = 40;
    end

    clc;
    close all;

    % 物理参数设置（保持原逻辑）
    variable_alpha_height=0.2; % um^-1
    variable_channel_height=6.5; % um
    variable_alpha_hgb=40.6; % M^-1um^-1
    variable_length_per_pixel=0.0236946; %um
    variable_area_per_pixel=0.0005614343; % um^2  每个像素的面积
    variable_hgb_molecular_mass=65000;
    variable_timestep=0.012; % 时间步长(秒)

    variable_temperature=300; %K
    constant_blotzmann=1.3807*10^-23; %J/K
    KbT=variable_temperature*constant_blotzmann;

    % 读取参考图像获取尺寸（保持原逻辑）
    pixel_dimension=double(imread('1.tif'));
    [m,n,o] = size(pixel_dimension);
    lx=m*variable_length_per_pixel;
    ly=n*variable_length_per_pixel;

    height_fix=-16500;
    hgb_fix=0;

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%   读取背景图像   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    background=double(imread('background.tif'));
    background_height = background(:,:,1); % 红色通道(无样本)
    background_green = background(:,:,2); % 绿色通道(无样本)
    background_hgb = background(:,:,3); % 蓝色通道(无样本)

    % 背景高度通道预处理
    min_background_height=min(background_height,[],'all');
    max_background_height=max(background_height,[],'all');
    mean_background_height=mean(background_height,'all');
    background_height=background_height-(min_background_height-1);
    multiplier_background_height=65535/max(background_height,[],'all');
    background_height=background_height.*multiplier_background_height;
    mean_background_height=mean(background_height,'all');

    % 背景血红蛋白通道预处理
    min_background_hgb=min(background_hgb,[],'all');
    max_background_hgb=max(background_hgb,[],'all');
    mean_background_hgb=mean(background_hgb,'all');
    background_hgb=background_hgb-(min_background_hgb-1);
    multiplier_background_hgb=65535/max(background_hgb,[],'all');
    background_hgb=background_hgb.*multiplier_background_hgb;
    mean_background_hgb=mean(background_hgb,'all');

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%   读取掩膜图像   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    mask_read_small=double(imread('mask.tif'));
    mask_small=mask_read_small(:,:,1);
    mask_small(mask_small > 0) = 1;

    mask_read_large=double(imread('mask3.tif'));
    mask_large=mask_read_large(:,:,1);
    mask_large(mask_large > 0) = 1;

    % ====================== 新增：计算mask3（mask_large）的面积 ======================
    % 1. 统计mask_large中有效像素数量（值为1的像素）
    mask3_valid_pixels = sum(mask_large(:) == 1);
    % 2. 计算总面积（有效像素数 × 每个像素的平方微米面积）
    mask3_total_area = mask3_valid_pixels * variable_area_per_pixel;
    % =================================================================================

    loop=total_frame/variable_averaging; % 使用传入的参数

    ct=1;
    fig_data_count=1;

    % 初始化累积变量
    pass_fft_mean_square_average=zeros(m,n);
    pass_cell_height_displacement_square_average_root=zeros(m,n);
    volume_data=zeros(total_frame,2);
    hgb_data=zeros(total_frame,2);

    % 初始化输出文件（先清空再写入，避免追加重复内容）
    output_file = 'deformation.txt';
    fileID = fopen(output_file, 'w');
    fprintf(fileID, '细胞变形分析结果\n');
    fprintf(fileID, '====================\n\n');
    fprintf(fileID, '当前参数：total_frame=%d, variable_averaging=%d\n\n', total_frame, variable_averaging);

    % 初始化体积和血红蛋白数据文件（清空历史数据）
    output_file_volume = 'volume_data_output.txt';
    fileID_volume = fopen(output_file_volume, 'w');
    fclose(fileID_volume);
    output_file_hgb = 'hgb_data_output.txt';
    fileID_hgb = fopen(output_file_hgb, 'w');
    fclose(fileID_hgb);

    for averaging=1:1:loop
        % 计算平均高度
        current_frame=1;
        cell_height_average=zeros(m,n);
        cell_height_cumulative=zeros(m,n);

        for k=ct:1:(variable_averaging*averaging)
            file_name = [num2str(k,'%d') '.tif'];  
            cell=double(imread(file_name));
            cell_height = cell(:,:,1); % 红色通道(有样本)

            % 高度通道预处理
            min_cell_height=min(cell_height,[],'all');
            cell_height=cell_height-(min_cell_height-1);
            multiplier_cell_height=65535/max(cell_height,[],'all');
            cell_height=cell_height.*multiplier_cell_height;

            % 计算相对高度
            height_relative= cell_height./(mean_background_height+height_fix);
            height_log=log10(height_relative);
            height_per_pixel=(height_log/(variable_alpha_height));
            height_per_pixel = filloutliers(height_per_pixel,'nearest','mean');

            % 应用大掩膜
            height_total_active_pixel=height_per_pixel.*mask_large;
            height_total_active_pixel(height_total_active_pixel < 0) = 0;
            cell_height_cumulative=cell_height_cumulative+height_total_active_pixel;

            current_frame=current_frame+1;
        end

        cell_height_average=cell_height_cumulative/(current_frame-1);

        % 计算位移和FFT
        cell_height_displacement_cumulative=zeros(m,n);
        cell_height_displacement_square_cumulative=zeros(m,n);
        fft_mean_square_cumulative=zeros(m,n);
        current_frame=1;

        for i=ct:1:(variable_averaging*averaging)
            file_name = [num2str(i,'%d') '.tif'];  
            cell=double(imread(file_name));
            cell_height = cell(:,:,1); % 红色通道
            cell_hgb = cell(:,:,3); % 蓝色通道

            % 高度通道预处理
            min_cell_height=min(cell_height,[],'all');
            cell_height=cell_height-(min_cell_height-1);
            multiplier_cell_height=65535/max(cell_height,[],'all');
            cell_height=cell_height.*multiplier_cell_height;

            % 血红蛋白通道预处理
            min_cell_hgb=min(cell_hgb,[],'all');
            cell_hgb=cell_hgb-(min_cell_hgb-1);
            multiplier_cell_hgb=65535/max(cell_hgb,[],'all');
            cell_hgb=cell_hgb.*multiplier_cell_hgb;

            % 计算高度与体积
            height_relative= cell_height./(mean_background_height+height_fix);
            height_log=log10(height_relative);
            height_per_pixel=(height_log/(variable_alpha_height));
            height_per_pixel = filloutliers(height_per_pixel,'nearest','mean');

            height_total_active_pixel=height_per_pixel;
            height_total_active_pixel(height_total_active_pixel < 0) = 0;
            height_total_active_pixel=height_total_active_pixel.*mask_large;
            volume_total = (sum(height_total_active_pixel*variable_area_per_pixel,'all'));

            % 保存体积数据（追加模式）
            fileID_volume = fopen(output_file_volume, 'a');
            volume_data(fig_data_count, 1) = fig_data_count;
            volume_data(fig_data_count, 2) = volume_total;
            fprintf(fileID_volume, 'Volume total at row %d: %.4f\n', fig_data_count, volume_data(fig_data_count, 2));
            fclose(fileID_volume);

            % 计算变形(位移)
            deformation_per_pixel= (height_total_active_pixel-cell_height_average).*mask_small.*1000; % 转为nm
            deformation_per_pixel = filloutliers(deformation_per_pixel,'nearest','mean');
            cell_height_displacement_square_cumulative=cell_height_displacement_square_cumulative+deformation_per_pixel.^2;

            % 计算FFT
            fft_deformation=fft2(double((deformation_per_pixel/1000000000)));
            fft_shift=fftshift(fft_deformation);
            fft_shift_abs=real(fft_shift);
            fft_shift_abs_square=(fft_shift_abs.^2);
            fft_2=(variable_area_per_pixel*(m*n)*(10^-12)).*((fft_shift_abs_square));
            fft_mean_square_cumulative=fft_mean_square_cumulative+fft_2;

            % 计算血红蛋白总量
            hgb_relative= cell_hgb./(mean_background_hgb+hgb_fix);
            hgb_log=log10(hgb_relative);
            hgb_per_pixel=(hgb_log.*variable_area_per_pixel.*variable_hgb_molecular_mass)./(-variable_alpha_hgb);
            hgb_per_pixel = filloutliers(hgb_per_pixel,'nearest','mean');

            hgb_total_active_pixel=hgb_per_pixel;
            hgb_total_active_pixel(hgb_total_active_pixel < 0) = 0;
            hgb_total_active_pixel=hgb_total_active_pixel.*mask_large;
            hgb_total = (sum(hgb_total_active_pixel,'all'))/1000;

            % 保存血红蛋白数据（追加模式）
            fileID_hgb = fopen(output_file_hgb, 'a');
            hgb_data(fig_data_count,1)=fig_data_count;
            hgb_data(fig_data_count,2)=hgb_total;
            fprintf(fileID_hgb, 'HGB total at row %d: %.4f\n', fig_data_count, hgb_data(fig_data_count, 2));
            fclose(fileID_hgb);

            current_frame=current_frame+1;
            fig_data_count=fig_data_count+1;
        end

        ct=ct+variable_averaging;
        
        % 计算平均位移和FFT
        cell_height_displacement_square_average=cell_height_displacement_square_cumulative/(current_frame-1);
        cell_height_displacement_square_average_root=sqrt(cell_height_displacement_square_average);
        cell_height_displacement_square_average_root=cell_height_displacement_square_average_root.*mask_small;
        
        fft_mean_square_average=(fft_mean_square_cumulative/(current_frame-1));
        
        pass_fft_mean_square_average=pass_fft_mean_square_average+fft_mean_square_average;
        pass_cell_height_displacement_square_average_root=pass_cell_height_displacement_square_average_root+cell_height_displacement_square_average_root;
    end

    % 最终计算结果
    final_fft_data=pass_fft_mean_square_average/loop;
    final_rms=pass_cell_height_displacement_square_average_root/loop;

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%   RMS位移统计计算   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % 提取有效区域的RMS值（仅掩膜内的像素）
    valid_rms = final_rms(mask_small > 0);

    % 计算统计值
    mean_rms = mean(valid_rms);          % 平均RMS位移
    std_rms = std(valid_rms);            % RMS位移的标准差
    max_rms = max(valid_rms);            % 最大RMS位移
    min_rms = min(valid_rms);            % 最小RMS位移
    sum_rms = sum(valid_rms);
    N_mask = numel(valid_rms);           % 有效像素数量
    average_rms = sum_rms / N_mask;

    % 写入RMS统计结果到文件
    fprintf(fileID, '1. RMS位移统计结果\n');
    fprintf(fileID, '   - 有效像素数量: %d\n', N_mask);
    fprintf(fileID, '   - 平均RMS位移 (Mean RMS displacement): %.6f nm\n', mean_rms);
    fprintf(fileID, '   - RMS位移标准差 (Standard deviation of RMS displacement): %.6f nm\n', std_rms);
    fprintf(fileID, '   - 最大RMS位移 (Max RMS displacement): %.6f nm\n', max_rms);
    fprintf(fileID, '   - 最小RMS位移 (Min RMS displacement): %.6f nm\n\n', min_rms);

    % ====================== 新增：写入mask3（mask_large）面积到文件 ======================
    fprintf(fileID, '1.5 mask3（大掩膜）面积统计结果\n');
    fprintf(fileID, '   - mask3有效像素数量: %d\n', mask3_valid_pixels);
    fprintf(fileID, '   - mask3总面积 (Total Area of mask3): %.6f μm²\n\n', mask3_total_area);
    % =================================================================================

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%   绘制平均RMS位移图   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    figure;
    surf(final_rms,'edgecolor','none');
    grid off
    axis off
    view(45, 75);
    daspect([1 1 1]);
    colormap(jet(256));
    shading interp
    axis square
    h2=colorbar;
    title({['Average Height Displacement']; ['Mean Root Square Mapping']},'FontSize',17,'fontweight','bold');
    title(h2,'nm')

    plot_scale_bar_2 = max(max_rms, abs(min_rms));
    caxis([0 plot_scale_bar_2]);

    txt = ['Root Mean Square of Displacement: ' num2str(average_rms) ' nm'];
    text(-0.2,-0.08,txt,'fontweight','bold','FontSize', 15, 'Units','normalized');
    set(gcf, 'Color', 'white');
    set(gca,'FontSize',15,'fontweight','bold');
    saveas(gcf, 'average_root_mean_square_map.png');

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%   频域处理   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % 调整矩阵为正方形
    if m~=n
        if m>n
            delete_column_number=m-n;
            final_fft_data = final_fft_data(1:end-delete_column_number,:);
        else 
            delete_row_number=n-m;
            final_fft_data = final_fft_data(:,1:end-delete_row_number);
        end
    end
    [m,n] = size(final_fft_data); % 重新获取调整后尺寸（去除多余的o维度）

    % 径向平均
    [Zr, R] = radialavg(final_fft_data,m/2,0,0);
    Zr=transpose(Zr);

    % 计算波矢
    qx_1=zeros(m,1);
    for k=0:m-1
        qx_1(k+1)=(2*pi/m)*(k);
    end
    qx_2 = fftshift(qx_1);
    qx_3 = unwrap(qx_2-2*pi);
    x_axis=qx_3/(variable_length_per_pixel*10^-6);

    qy_1=zeros(n,1);
    for k=0:n-1
        qy_1(k+1)=(2*pi/n)*(k);
    end
    qy_2 = fftshift(qy_1);
    qy_3 = unwrap(qy_2-2*pi);
    y_axis=qy_3/(variable_length_per_pixel*10^-6);

    % 处理x轴波矢（取正半部分）
    if mod(m, 2) == 0
      x_axis_2d=x_axis((m/2)+1:end,:);
    else
      x_axis_2d=x_axis((m+1)/2:end,:);
    end

    % 写入频域数据到文件
    fprintf(fileID, '2. 频域均方位移数据\n');
    fprintf(fileID, '   - 频域数据最大值: %.6e μm^4\n', max(final_fft_data(:)));
    fprintf(fileID, '   - 频域数据最小值: %.6e μm^4\n', min(final_fft_data(:)));
    fprintf(fileID, '   - 波矢范围: %.6f 至 %.6f rad/μm\n\n', min(x_axis), max(x_axis));

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%   绘制频域分布图   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    figure('Name', 'Log of mean square displacement');
    surf(y_axis,x_axis,final_fft_data,'edgecolor','none');
    set(gca,'ZScale','log');
    colormap('jet')
    h2=colorbar;
    set(gca,'ColorScale','log')
    title({['Log of mean square displacement (h) vs']; [' Wave vector (q) in frequency domain']},'FontSize',15,'fontweight','bold');
    title(h2,'\mum^{4}')
    set(gca,'fontweight','bold','fontsize',12)
    xlabel('Wave vector (rad/\mum)','Units','normalized','FontWeight','bold',...
        'FontSize',12,'Rotation',21);
    xh = get(gca,'XLabel');
    set(xh, 'Units', 'Normalized')
    pos = get(xh, 'Position');
    set(xh, 'Position',pos.*[1.15,0.2,1.2],'Rotation',18)
    ylabel('Wave vector (rad/\mum)','Units','normalized','FontWeight','bold',...
        'FontSize',12,'Rotation',-33);
    yh = get(gca,'YLabel');
    set(yh, 'Units', 'Normalized')
    pos = get(yh, 'Position');
    set(yh, 'Position',pos.*[0.25,-0.3,0.1],'Rotation',-32)
    zlabel({['Mean square displacement']; ['\Delta h^{2} (\mum^{4})']},'FontSize',12,'fontweight','bold') 
    saveas(gcf, 'Log_of_mean_square_displacement_Distribution.png');

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%   绘制径向平均图   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    figure('Name', 'Hemoglobin Mass Mapping1');
    scatter(x_axis_2d,Zr,'.');
    set(gca,'XScale','log');
    set(gca,'YScale','log');
    title({['Radial average of mean square displacement (h) vs']; [' Wave vector (q) in frequency domain (Log Scale)']},'FontSize',15,'fontweight','bold');
    xlabel('Wave vector (rad/\mum)','FontSize',12,'fontweight','bold') 
    ylabel('Mean square displacement \Delta h^{2} (\mum^{4})','FontSize',12,'fontweight','bold');
    set(gca,'fontweight','bold','fontsize',12)
    saveas(gcf, 'Log_of_mean_square_displacement_vs_Wave_vector.png');

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%   拟合理论模型   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    axis_length = length(x_axis_2d);
    kappa = 1*10^-20; % 弯曲模量 (J)
    sigma = 1*10^-6;  % 张力系数 (J/m²)

    u_fit = zeros(axis_length, 1);
    u_analytical = zeros(axis_length, 1);

    % 修复min函数调用：多参数需放入数组
    max_count = min([axis_length, size(x_axis_2d, 1), length(Zr)]);

    for count = 1:max_count
        u_fit(count) = Zr(count);
        u_analytical(count) = (KbT * 1e12) / (kappa * x_axis_2d(count)^4 + sigma * x_axis_2d(count)^2);
    end

    % 写入拟合数据到文件
    fprintf(fileID, '3. 径向平均与拟合数据\n');
    fprintf(fileID, '   - 理论模型张力系数(sigma): %.6e J/m²\n', sigma);
    fprintf(fileID, '   - 理论模型弯曲模量(kappa): %.6e J\n', kappa);
    fprintf(fileID, '   - 径向平均数据点数量: %d\n\n', max_count);

    % 写入波矢-均方位移详细数据
    fprintf(fileID, '4. 波矢-均方位移详细数据(对数坐标)\n');
    fprintf(fileID, '   波矢(rad/μm)     均方位移(μm⁴)     理论拟合值(μm⁴)\n');
    for i = 1:max_count
        fprintf(fileID, '   %.6e     %.6e     %.6e\n', x_axis_2d(i), Zr(i), u_analytical(i));
    end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%   读取hgb和volume数据并计算平均值   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % -------------------------- 血红蛋白数据处理 --------------------------
    fileID_hgb_read = fopen(output_file_hgb, 'r');
    if fileID_hgb_read == -1
        fprintf(fileID, '\n5. 血红蛋白数据读取错误\n');
        fprintf(fileID, '   - 无法打开文件: %s\n', output_file_hgb);
    else
        hgb_values = [];
        while ~feof(fileID_hgb_read)
            line = fgetl(fileID_hgb_read);
            if ~isempty(line)
                % 匹配 "HGB total at row X: Y" 格式，提取Y值
                tokens = regexp(line, 'HGB total at row \d+: (\d+\.\d+)', 'tokens');
                if ~isempty(tokens{1})
                    hgb_val = str2double(tokens{1}{1});
                    hgb_values = [hgb_values; hgb_val];
                end
            end
        end
        fclose(fileID_hgb_read);
        
        % 计算血红蛋白统计值并写入
        if ~isempty(hgb_values)
            mean_hgb = mean(hgb_values);
            std_hgb = std(hgb_values);
            max_hgb = max(hgb_values);
            min_hgb = min(hgb_values);
            fprintf(fileID, '\n5. 血红蛋白(HGB)统计结果\n');
            fprintf(fileID, '   - 数据点数量: %d\n', length(hgb_values));
            fprintf(fileID, '   - 平均血红蛋白总量: %.6f\n', mean_hgb);
            fprintf(fileID, '   - 血红蛋白标准差: %.6f\n', std_hgb);
            fprintf(fileID, '   - 最大血红蛋白总量: %.6f\n', max_hgb);
            fprintf(fileID, '   - 最小血红蛋白总量: %.6f\n', min_hgb);
        else
            fprintf(fileID, '\n5. 血红蛋白数据解析错误\n');
            fprintf(fileID, '   - 文件无有效数据: %s\n', output_file_hgb);
        end
    end

    % -------------------------- 体积数据处理 --------------------------
    fileID_volume_read = fopen(output_file_volume, 'r');
    if fileID_volume_read == -1
        fprintf(fileID, '\n6. 体积数据读取错误\n');
        fprintf(fileID, '   - 无法打开文件: %s\n', output_file_volume);
    else
        volume_values = [];
        while ~feof(fileID_volume_read)
            line = fgetl(fileID_volume_read);
            if ~isempty(line)
                % 匹配 "Volume total at row X: Y" 格式，提取Y值
                tokens = regexp(line, 'Volume total at row \d+: (\d+\.\d+)', 'tokens');
                if ~isempty(tokens{1})
                    volume_val = str2double(tokens{1}{1});
                    volume_values = [volume_values; volume_val];
                end
            end
        end
        fclose(fileID_volume_read);
        
        % 计算体积统计值并写入
        if ~isempty(volume_values)
            mean_volume = mean(volume_values);
            std_volume = std(volume_values);
            max_volume = max(volume_values);
            min_volume = min(volume_values);
            fprintf(fileID, '\n6. 体积(Volume)统计结果\n');
            fprintf(fileID, '   - 数据点数量: %d\n', length(volume_values));
            fprintf(fileID, '   - 平均体积: %.6f μm³\n', mean_volume);
            fprintf(fileID, '   - 体积标准差: %.6f μm³\n', std_volume);
            fprintf(fileID, '   - 最大体积: %.6f μm³\n', max_volume);
            fprintf(fileID, '   - 最小体积: %.6f μm³\n', min_volume);
        else
            fprintf(fileID, '\n6. 体积数据解析错误\n');
            fprintf(fileID, '   - 文件无有效数据: %s\n', output_file_volume);
        end
    end

    % 关闭主输出文件
    fclose(fileID);

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%   绘制拟合图   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    figure('Name', 'Fitting');
    x_axis_fit = x_axis_2d(1:max_count);
    plot(x_axis_fit, u_analytical(1:max_count), 'b-', 'LineWidth', 1.5, 'DisplayName', '理论拟合');
    hold on;
    plot(x_axis_fit, u_fit(1:max_count), 'ro', 'MarkerSize', 5, 'MarkerFaceColor', 'r', 'DisplayName', '实验数据');
    set(gca,'XScale','log');
    set(gca,'YScale','log');
    title({['Radial average of mean square displacement (h) vs']; [' Wave vector (q) in frequency domain (Log Scale)']},'FontSize',16,'fontweight','bold');
    xlabel('Wave vector (rad/\mum)','FontSize',12,'fontweight','bold') 
    ylabel('Mean square displacement \Delta h^{2} (\mum^{2})','FontSize',13,'fontweight','bold');
    legend('Location','best');
    txt = ['Tension Coefficient: ' num2str(sigma) ' J/m²'];
    text(0.04, 0.09, txt, 'Units', 'normalized', 'FontWeight', 'bold', 'FontSize', 12);
    saveas(gcf, 'Log_of_mean_square_fitting.png');

    % 保存中间变量（新增mask3面积相关变量，便于后续分析）
    save('mat_values.mat', 'final_fft_data', 'final_rms', 'volume_data', 'hgb_data', 'Zr', 'x_axis_2d', 'u_analytical', 'u_fit', 'mask3_valid_pixels', 'mask3_total_area');
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%   径向平均函数   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [Zr, R] = radialavg(z,m,xo,yo)
    if ~exist('xo','var')
        xo = 0;
    end
    if ~exist('yo','var')
        yo = 0;
    end

    N = size(z,1);
    [X,Y] = meshgrid(-1:2/(N-1):1);
    X = X - xo;
    Y = Y - yo;

    r = sqrt(X.^2 + Y.^2);

    dr = 1/(m-1);
    rbins = linspace(-dr/2, 1+dr/2, m+1);

    R = (rbins(1:end-1) + rbins(2:end))/2;

    Zr = zeros(1,m);
    nans = ~isnan(z);

    for j=1:m-1
        bins = r >= rbins(j) & r < rbins(j+1);
        bins = logical(bins .* nans);
        n = sum(sum(bins));
        if n ~= 0
            Zr(j) = sum(z(bins))/n;
        else
            Zr(j) = NaN;
        end
    end

    % 处理最后一个bin
    bins = r >= rbins(m) & r <= 1;
    bins = logical(bins .* nans);
    n = sum(sum(bins));
    if n ~= 0
        Zr(m) = sum(z(bins))/n;
    else
        Zr(m) = NaN;
    end
end