clc;
clear;
close all;

% 设置根目录（每个子文件夹为一个数据集）
rootDir = "G:\NORMAL\ln\ln0.1\ln1\channel_5=0.55\4_background2";  % 替换为实际根目录
dataFolders = dir(rootDir);
dataFolders = dataFolders([dataFolders.isdir] & ~ismember({dataFolders.name}, {'.','..'}));

originalDir = pwd;
filesToCopy = {'stiffness_full_final_with_fitting_fft_edited.m'};

for i = 1:length(dataFolders)
    datasetPath = fullfile(rootDir, dataFolders(i).name);
    fprintf('Processing dataset: %s\n', datasetPath);
    
    % 检查掩码文件
    mask1 = fullfile(datasetPath, 'mask.tif');
    mask2 = fullfile(datasetPath, 'mask2.tif');
    mask3 = fullfile(datasetPath, 'mask3.tif');
    if ~(exist(mask1, 'file')==2 && exist(mask2, 'file')==2 && exist(mask3, 'file')==2)
        fprintf('%s: 无掩码，跳过\n', datasetPath);
        continue;
    end
    
    % 检查是否已处理
    deformationFile = fullfile(datasetPath, 'deformation.txt');
    if exist(deformationFile, 'file')==2
        fprintf('%s: 已处理，跳过\n', datasetPath);
        continue;
    end
    
    % 复制处理脚本到当前数据集文件夹
    for k = 1:length(filesToCopy)
        sourceFile = fullfile(originalDir, filesToCopy{k});
        destinationFile = fullfile(datasetPath, filesToCopy{k});
        copyfile(sourceFile, destinationFile);
    end
    
    % 统计当前文件夹内的有序图片数量（以 .tif 为例，可根据实际格式调整）
    imgFiles = dir(fullfile(datasetPath, '*.tif'));
    imgFiles = {imgFiles.name};
    % 筛选有序图片（如 1.tif, 2.tif... 或 13.tif,14.tif...）
    numPattern = '^\d+\.tif$';
    validImgs = regexp(imgFiles, numPattern, 'match');
    validImgs = ~cellfun('isempty', validImgs);
    total_frame = sum(validImgs);
    variable_averaging = total_frame;  % 与 total_frame 保持一致
    
    fprintf('  图片数量: %d, total_frame=%d, variable_averaging=%d\n', ...
        total_frame, total_frame, variable_averaging);
    
    % 切换到数据集文件夹并运行脚本，传入参数
    cd(datasetPath);
    run(['stiffness_full_final_with_fitting_fft_edited(' num2str(total_frame) ',' num2str(variable_averaging) ')']);
    
    fclose('all');
    cd(originalDir);
end

disp('所有数据集处理完成！');