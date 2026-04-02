%% Code generation for Depth-Anything-V2-Small
% Uses MATLAB Coder Support Package for PyTorch to generate C++ from .pt2

%% Configuration
outDir = fullfile(pwd, 'codegen_out');
if ~exist(outDir, 'dir')
    mkdir(outDir);
end

%% Coder configuration
cfg = coder.config('lib', 'ecoder', true);
cfg.TargetLang = 'C++';
cfg.GenerateReport = true;
cfg.GenCodeOnly = true;
cfg.EnableOpenMP = true;
cfg.SupportNonFinite = true;

% Deep Learning configuration - no external library
dlcfg = coder.DeepLearningConfig('none');
cfg.DeepLearningConfig = dlcfg;

%% Input type specification
% Input: [1, 3, 518, 784] single (NCHW format)
inputType = coder.typeof(single(0), [1 3 518 784], [false false false false]);

%% Run code generation
fprintf('Starting code generation...\n');
tic;
codegen -config cfg depth_infer -args {inputType} -d codegen_out -report
t = toc;
fprintf('Code generation completed in %.1f seconds.\n', t);
fprintf('Output directory: %s\n', outDir);
