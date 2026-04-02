%% Benchmark MATLAB Coder generated code for Depth-Anything-V2-Small

%% Build MEX via codegen
fprintf('Building MEX via codegen...\n');
cfg = coder.config('mex');
cfg.EnableOpenMP = false;
inputType = coder.typeof(single(0), [1 3 518 784], [false false false false]);

if exist('codegen_mex', 'dir')
    rmdir('codegen_mex', 's');
end
codegen -config cfg depth_infer -args {inputType} -d codegen_mex -report
fprintf('MEX codegen complete.\n');

%% Load reference data
fprintf('\nLoading reference data...\n');
fid = fopen('../reference/input.bin', 'r');
pt_input_raw = fread(fid, 1*3*518*784, 'single');
fclose(fid);
% PyTorch stores as row-major NCHW: fastest-varying is W, then H, then C, then N
pt_input = reshape(pt_input_raw, [784, 518, 3, 1]);
pt_input = single(permute(pt_input, [4, 3, 2, 1]));  % -> [1, 3, 518, 784] MATLAB col-major

fid = fopen('../reference/output.bin', 'r');
pt_output_raw = fread(fid, 1*518*784, 'single');
fclose(fid);
pt_output = reshape(pt_output_raw, [784, 518, 1]);
pt_output = permute(pt_output, [3, 2, 1]);  % -> [1, 518, 784]

%% Run inference
fprintf('\nRunning inference via MEX...\n');
tic;
ml_output = depth_infer_mex(pt_input);
t_first = toc;
fprintf('First call (with init): %.1f ms\n', t_first * 1000);
ml_output = reshape(ml_output, [1, 518, 784]);

%% Accuracy
err = abs(double(ml_output) - double(pt_output));
fprintf('\n=== Accuracy Comparison vs PyTorch Reference ===\n');
fprintf('Reference range: [%.4f, %.4f]\n', min(pt_output(:)), max(pt_output(:)));
fprintf('MATLAB Coder:    [%.4f, %.4f]\n', min(ml_output(:)), max(ml_output(:)));
fprintf('Max abs error:   %.6e\n', max(err(:)));
fprintf('Mean abs error:  %.6e\n', mean(err(:)));
rmse_val = sqrt(mean(err(:).^2));
ref_range = max(pt_output(:)) - min(pt_output(:));
fprintf('RMSE:            %.6e\n', rmse_val);
fprintf('Relative RMSE:   %.6e\n', rmse_val / ref_range);

%% Benchmark
fprintf('\n=== Benchmark (5 runs) ===\n');
times = zeros(1, 5);
for i = 1:5
    tic;
    ml_output = depth_infer_mex(pt_input);
    times(i) = toc * 1000;
    fprintf('Run %d: %.1f ms\n', i, times(i));
end
fprintf('Average: %.1f ms\n', mean(times));

%% Count generated code metrics
cppFile = fullfile('codegen_out', 'depth_infer.cpp');
fid = fopen(cppFile, 'r');
content = fread(fid, '*char')';
fclose(fid);
nlines = numel(strfind(content, char(10))) + 1;
binFiles = dir(fullfile('codegen_out', '*.bin'));
nBins = numel(binFiles);
totalBinBytes = sum([binFiles.bytes]);

%% Save results
fid = fopen('matlab_coder_results.json', 'w');
fprintf(fid, '{\n');
fprintf(fid, '  "approach": "MATLAB Coder",\n');
fprintf(fid, '  "first_call_ms": %.1f,\n', t_first * 1000);
fprintf(fid, '  "avg_inference_ms": %.1f,\n', mean(times));
fprintf(fid, '  "std_inference_ms": %.1f,\n', std(times));
fprintf(fid, '  "max_abs_error": %.6e,\n', max(err(:)));
fprintf(fid, '  "mean_abs_error": %.6e,\n', mean(err(:)));
fprintf(fid, '  "rmse": %.6e,\n', rmse_val);
fprintf(fid, '  "relative_rmse": %.6e,\n', rmse_val / ref_range);
fprintf(fid, '  "generated_cpp_lines": %d,\n', nlines);
fprintf(fid, '  "weight_files": %d,\n', nBins);
fprintf(fid, '  "total_weight_size_mb": %.1f\n', totalBinBytes / 1024 / 1024);
fprintf(fid, '}\n');
fclose(fid);
fprintf('\nResults saved to matlab_coder_results.json\n');
