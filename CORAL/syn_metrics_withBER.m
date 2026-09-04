%{
========================================================================================
     CORAL UDA NTN Channel Equalization, BER, MSE, NMSE & SSIM Performance Evaluation
========================================================================================
OVERVIEW:
  This script evaluates the transmission, equalization, BER, MSE, NMSE, and SSIM
  performance of CORAL UDA channel estimation models across SNR subfolders (e.g. LS_-10, LS_-5, ...).
  
  For each SNR subfolder, both Source Domain and Target Domain test sets are evaluated:
    - Source Domain: loaded from 'testChannel_source.mat'
    - Target Domain: loaded from 'testChannel_target.mat'

  In each test channel file, the variables are:
    - H_perfect_test : Ground truth channel grid (Nsamples x 132 x 14)
    - H_LI_test      : Linear interpolation baseline (Nsamples x 132 x 14)
    - H_output_test  : Machine learning model estimated channel (Nsamples x 132 x 14)
    - H_LS_test      : Sparse LS estimates at pilot locations (Nsamples x 88)
    - pilot_rows     : Subcarrier row indices of pilot elements (1-indexed for MATLAB)
    - pilot_cols     : OFDM symbol column indices of pilot elements (1-indexed for MATLAB)
    - snr            : Subfolder SNR value (dB)

  The script:
    1. Computes MMSE benchmark using empirical covariance R_hh from H_perfect_test and H_LS_test.
    2. Simulates 5G NR PDSCH OFDM transmission over H_perfect_test + AWGN.
    3. Performs MMSE equalization & demodulation with H_LI, H_output (Infer), and H_MMSE.
    4. Computes BER, MSE, NMSE, NMSE (dB), and SSIM across all SNR points for BOTH Source and Target.
    5. Saves per-SNR results inside each SNR subfolder.
    6. Plots and saves figures separately for Source and Target (plus joint comparison).
    7. Saves domain-specific synthesized MAT files ('synthesized_results_source.mat',
       'synthesized_results_target.mat') and a unified 'synthesized_results.mat'.

USAGE:
  syn_metrics_withBER()                                    % Evaluates default batch folder
  syn_metrics_withBER('batch_folder_path')                 % Evaluates specific batch folder
  syn_metrics_withBER('batch_folder_path', 'Custom Label') % Evaluates with custom label
  syn_metrics_withBER('batch_folder_path', 'Custom Label', {'layer1', 'layer1_layer2'})
========================================================================================
%}

function results = syn_metrics_withBER(batch_folder, labelname, extractLayer)
    if exist('mfilename', 'builtin') && ~isempty(mfilename('fullpath'))
        script_dir = fileparts(mfilename('fullpath'));
    else
        script_dir = pwd;
    end

    % Default configuration fallbacks
    batch_folder_ = 'C:\Users\AT30890\Hoctap\1_Hprediction\working\H_predict_NTN\Hest_NTN_UDA_clean\CORAL\A100__DUR100_2p18e9_600km_30kHz\LS_Attention_standardize';
    labelname_ = 'LS+Attention(Std) Inferred';
    extractLayer_ = {'layer1', 'layer1_layer2'};

    if nargin < 1 || isempty(batch_folder)
        batch_folder = batch_folder_;
    end
    if nargin < 2 || isempty(labelname)
        labelname = labelname_;
    end
    if nargin < 3 || isempty(extractLayer)
        extractLayer = extractLayer_;
    end

    % Add paths for helper functions
    if exist(fullfile(script_dir, '..', 'single_dataset', 'helper'), 'dir')
        addpath(fullfile(script_dir, '..', 'single_dataset', 'helper'));
    end
    if exist(fullfile(script_dir, '..', 'JMMD', 'helper'), 'dir')
        addpath(fullfile(script_dir, '..', 'JMMD', 'helper'));
    end
    addpath(fullfile(script_dir, '..'));

    % Check if batch_folder directly contains SNR subfolders with testChannel_*.mat
    direct_snr = discover_snr_folders(batch_folder);

    if isempty(direct_snr)
        % Search child directories (e.g. layer1, layer1_layer2) for target batch folders
        batch_targets = {};
        
        % First check if any requested extractLayer directories exist
        if iscell(extractLayer) && ~isempty(extractLayer)
            for el_idx = 1:length(extractLayer)
                candidate_layer = fullfile(batch_folder, extractLayer{el_idx});
                if exist(candidate_layer, 'dir') && ~isempty(discover_snr_folders(candidate_layer))
                    batch_targets{end+1} = candidate_layer; %#ok<AGROW>
                end
            end
        end

        % If not found via extractLayer, scan all child directories
        if isempty(batch_targets)
            dir_items = dir(batch_folder);
            for k = 1:length(dir_items)
                if dir_items(k).isdir && ~strcmp(dir_items(k).name, '.') && ~strcmp(dir_items(k).name, '..')
                    sub_path = fullfile(batch_folder, dir_items(k).name);
                    if ~isempty(discover_snr_folders(sub_path))
                        batch_targets{end+1} = sub_path; %#ok<AGROW>
                    end
                end
            end
        end

        if isempty(batch_targets)
            error('No subfolders containing testChannel_*.mat or inferredChannel.mat found in:\n  %s', batch_folder);
        end

        fprintf('Found %d batch target folder(s) to process:\n', length(batch_targets));
        for b = 1:length(batch_targets)
            fprintf('  [%d] %s\n', b, batch_targets{b});
        end
        fprintf('\n');

        results = cell(1, length(batch_targets));
        for t_idx = 1:length(batch_targets)
            [~, leaf_name] = fileparts(batch_targets{t_idx});
            layer_label = sprintf('%s (%s)', labelname, leaf_name);
            fprintf('========================================================================\n');
            fprintf('>>> Processing Batch Target [%d/%d]: %s <<<\n', t_idx, length(batch_targets), batch_targets{t_idx});
            fprintf('========================================================================\n');
            results{t_idx} = process_single_batch(batch_targets{t_idx}, layer_label);
        end
    else
        results = process_single_batch(batch_folder, labelname);
    end
end

%% Helper function to discover SNR subfolders in a target directory
function snr_subfolders = discover_snr_folders(target_dir)
    snr_subfolders = {};
    if ~exist(target_dir, 'dir')
        return;
    end
    dir_items = dir(target_dir);
    for k = 1:length(dir_items)
        if dir_items(k).isdir && ~strcmp(dir_items(k).name, '.') && ~strcmp(dir_items(k).name, '..')
            sub_path = fullfile(target_dir, dir_items(k).name);
            has_target = exist(fullfile(sub_path, 'testChannel_target.mat'), 'file');
            has_source = exist(fullfile(sub_path, 'testChannel_source.mat'), 'file');
            has_inferred = exist(fullfile(sub_path, 'inferredChannel.mat'), 'file');
            if has_target || has_source || has_inferred
                snr_subfolders{end+1} = sub_path; %#ok<AGROW>
            end
        end
    end
end

%% Primary evaluation function for a single batch folder
function combined_results = process_single_batch(batch_folder, labelname)
    if nargin < 2 || isempty(labelname)
        labelname = 'LS+Attention(Std) Inferred';
    end

    fprintf('\n========================================================================\n');
    fprintf('    CORAL UDA Evaluation from Test Channels (Source & Target Domains)   \n');
    fprintf('========================================================================\n');
    fprintf('Batch Folder: %s\n', batch_folder);
    fprintf('Plot Label  : %s\n\n', labelname);

    % Discover & parse SNR subfolders
    dir_items = dir(batch_folder);
    snr_subfolders = {};
    snr_vals = [];

    for k = 1:length(dir_items)
        if dir_items(k).isdir && ~strcmp(dir_items(k).name, '.') && ~strcmp(dir_items(k).name, '..')
            subname = dir_items(k).name;
            sub_path = fullfile(batch_folder, subname);
            
            % Locate sample mat file to read true SNR value
            sample_file = '';
            if exist(fullfile(sub_path, 'testChannel_target.mat'), 'file')
                sample_file = fullfile(sub_path, 'testChannel_target.mat');
            elseif exist(fullfile(sub_path, 'testChannel_source.mat'), 'file')
                sample_file = fullfile(sub_path, 'testChannel_source.mat');
            elseif exist(fullfile(sub_path, 'inferredChannel.mat'), 'file')
                sample_file = fullfile(sub_path, 'inferredChannel.mat');
            end

            if ~isempty(sample_file)
                mat_data = load(sample_file, 'snr', 'SNRdB');
                if isfield(mat_data, 'snr') && ~isempty(mat_data.snr)
                    snr_val = double(mat_data.snr);
                elseif isfield(mat_data, 'SNRdB') && ~isempty(mat_data.SNRdB)
                    snr_val = double(mat_data.SNRdB);
                else
                    % Extract numeric SNR from folder name (e.g. 'LS_-10', 'SNR_5dB', '5')
                    tokens = regexp(subname, '(-?\d+(\.\d+)?)', 'tokens');
                    if ~isempty(tokens)
                        snr_val = str2double(tokens{end}{1});
                    else
                        continue;
                    end
                end
                snr_subfolders{end+1} = sub_path; %#ok<AGROW>
                snr_vals(end+1) = snr_val; %#ok<AGROW>
            end
        end
    end

    if isempty(snr_subfolders)
        error('No valid SNR subfolders found in:\n  %s', batch_folder);
    end

    % Sort subfolders by SNR in ascending order
    [SNRdB_sorted, sort_idx] = sort(snr_vals);
    snr_subfolders = snr_subfolders(sort_idx);
    num_snr = length(SNRdB_sorted);

    fprintf('Discovered %d SNR subfolders (SNRdB = %s):\n', num_snr, mat2str(SNRdB_sorted));
    for i = 1:num_snr
        [~, sf_name] = fileparts(snr_subfolders{i});
        fprintf('  [%d] %s -> SNRdB = %.1f dB\n', i, sf_name, SNRdB_sorted(i));
    end
    fprintf('\n');

    % Domains to evaluate
    domains = {'source', 'target'};
    domain_results = struct();

    for d_idx = 1:length(domains)
        domain = domains{d_idx};
        fprintf('------------------------------------------------------------------------\n');
        fprintf('>>> Evaluating Domain: %s <<<\n', upper(domain));
        fprintf('------------------------------------------------------------------------\n');

        domain_res = evaluate_domain(batch_folder, snr_subfolders, SNRdB_sorted, domain, labelname);
        domain_results.(domain) = domain_res;
    end

    % Save combined / unified synthesized results
    combined_results = struct();
    % Top-level target fields for full compatibility with syn_syn_compare_multiModels
    if isfield(domain_results, 'target') && ~isempty(domain_results.target)
        combined_results = domain_results.target;
    end
    combined_results.source = domain_results.source;
    combined_results.target = domain_results.target;
    combined_results.SNRdB  = SNRdB_sorted;

    unified_mat_path = fullfile(batch_folder, 'synthesized_results.mat');
    save(unified_mat_path, '-struct', 'combined_results');
    fprintf('\n[Saved] Unified synthesized MAT file -> %s\n', unified_mat_path);

    % Generate side-by-side Source vs Target comparison figures
    if isfield(domain_results, 'source') && isfield(domain_results, 'target') && ...
       ~isempty(domain_results.source) && ~isempty(domain_results.target)
        plot_source_vs_target_figures(batch_folder, SNRdB_sorted, domain_results.source, domain_results.target, labelname);
    end

    % Generate unified markdown summary report
    generate_unified_markdown_report(batch_folder, SNRdB_sorted, domain_results, labelname);

    fprintf('\n========================================================================\n');
    fprintf(' Evaluation Completed for Batch Folder: %s\n', batch_folder);
    fprintf('========================================================================\n\n');
end

%% Domain-specific evaluation function
function save_struct = evaluate_domain(batch_folder, snr_subfolders, SNRdB_sorted, domain, labelname)
    num_snr = length(SNRdB_sorted);
    domain_title = [upper(domain(1)), domain(2:end)];

    % 5G NR Carrier & PDSCH Configuration (30 kHz SCS, 11 RBs = 132 subcarriers, 14 symbols)
    carrier = nrCarrierConfig;
    carrier.SubcarrierSpacing = 30;     % 30 kHz SCS
    carrier.NSizeGrid = 11;             % 11 RBs (132 subcarriers)
    carrier.CyclicPrefix = 'Normal';

    nSubcarriers = carrier.NSizeGrid * 12;
    nSymbols     = carrier.SymbolsPerSlot;

    pdsch = nrPDSCHConfig;
    pdsch.PRBSet = 0:carrier.NSizeGrid-1;
    pdsch.SymbolAllocation = [0, carrier.SymbolsPerSlot];
    pdsch.MappingType = 'A';
    pdsch.NID = carrier.NCellID;
    pdsch.RNTI = 1;
    pdsch.VRBToPRBInterleaving = 0;
    pdsch.VRBBundleSize = 4;
    pdsch.NumLayers = 1;
    pdsch.Modulation = '16QAM';

    pdsch.DMRS.DMRSPortSet = [];
    pdsch.DMRS.DMRSTypeAPosition = 2;
    pdsch.DMRS.DMRSLength = 1;
    pdsch.DMRS.DMRSAdditionalPosition = 1;
    pdsch.DMRS.DMRSConfigurationType = 2;
    pdsch.DMRS.NumCDMGroupsWithoutData = 1;
    pdsch.DMRS.NIDNSCID = 1;
    pdsch.DMRS.NSCID = 0;

    [pdschIndices, pdschInfo] = nrPDSCHIndices(carrier, pdsch);
    dmrsSymbols = nrPDSCHDMRS(carrier, pdsch);
    dmrsIndices = nrPDSCHDMRSIndices(carrier, pdsch);

    % Metric storage vectors across SNR points
    ber_li_arr       = zeros(1, num_snr);
    ber_infer_arr    = zeros(1, num_snr);
    ber_mmse_arr     = zeros(1, num_snr);

    mse_li_arr       = zeros(1, num_snr);
    mse_infer_arr    = zeros(1, num_snr);
    mse_mmse_arr     = zeros(1, num_snr);

    nmse_li_arr      = zeros(1, num_snr);
    nmse_infer_arr   = zeros(1, num_snr);
    nmse_mmse_arr    = zeros(1, num_snr);

    nmse_li_db_arr   = zeros(1, num_snr);
    nmse_infer_db_arr= zeros(1, num_snr);
    nmse_mmse_db_arr = zeros(1, num_snr);

    ssim_li_arr      = zeros(1, num_snr);
    ssim_infer_arr   = zeros(1, num_snr);
    ssim_mmse_arr    = zeros(1, num_snr);

    % Loop over SNR subfolders
    for s_idx = 1:num_snr
        snr_val = SNRdB_sorted(s_idx);
        sf_path = snr_subfolders{s_idx};

        % Determine target file: testChannel_<domain>.mat (or fallback to inferredChannel.mat)
        mat_filename = sprintf('testChannel_%s.mat', domain);
        mat_path = fullfile(sf_path, mat_filename);

        if ~exist(mat_path, 'file')
            % Fallback check
            alt_mat = fullfile(sf_path, 'inferredChannel.mat');
            if exist(alt_mat, 'file')
                mat_path = alt_mat;
            else
                error('Missing test channel file for %s domain:\n  %s', domain, mat_path);
            end
        end

        data = load(mat_path);

        % 1. Extract ground truth H_perfect [numUE x 132 x 14]
        if isfield(data, 'H_perfect_test')
            H_perfect_raw = align_channel_matrix(data.H_perfect_test, nSubcarriers, nSymbols);
        elseif isfield(data, 'H_perfect')
            H_perfect_raw = align_channel_matrix(data.H_perfect, nSubcarriers, nSymbols);
        else
            error('H_perfect_test or H_perfect not found in %s', mat_path);
        end

        % 2. Extract linear interpolation baseline H_li [numUE x 132 x 14]
        if isfield(data, 'H_LI_test')
            H_li_raw = align_channel_matrix(data.H_LI_test, nSubcarriers, nSymbols);
        elseif isfield(data, 'H_li_test')
            H_li_raw = align_channel_matrix(data.H_li_test, nSubcarriers, nSymbols);
        elseif isfield(data, 'H_li')
            H_li_raw = align_channel_matrix(data.H_li, nSubcarriers, nSymbols);
        elseif isfield(data, 'H_LI')
            H_li_raw = align_channel_matrix(data.H_LI, nSubcarriers, nSymbols);
        elseif isfield(data, 'H_ls')
            H_li_raw = align_channel_matrix(data.H_ls, nSubcarriers, nSymbols);
        else
            H_li_raw = H_perfect_raw;
        end

        % 3. Extract model inferred output H_infer [numUE x 132 x 14]
        if isfield(data, 'H_output_test')
            H_infer_raw = align_channel_matrix(double(data.H_output_test), nSubcarriers, nSymbols);
        elseif isfield(data, 'H_output')
            H_infer_raw = align_channel_matrix(double(data.H_output), nSubcarriers, nSymbols);
        elseif isfield(data, 'H_LS_infer')
            H_infer_raw = align_channel_matrix(double(data.H_LS_infer), nSubcarriers, nSymbols);
        elseif isfield(data, 'H_li_infer')
            H_infer_raw = align_channel_matrix(double(data.H_li_infer), nSubcarriers, nSymbols);
        elseif isfield(data, 'H_LI_infer')
            H_infer_raw = align_channel_matrix(double(data.H_LI_infer), nSubcarriers, nSymbols);
        elseif isfield(data, 'H_infer')
            H_infer_raw = align_channel_matrix(double(data.H_infer), nSubcarriers, nSymbols);
        else
            error('No inferred channel field (H_output_test, H_LS_infer, or H_infer) found in %s', mat_path);
        end

        % 4. Extract Pilot Coordinates / Linear Indices
        if isfield(data, 'pilot_rows') && isfield(data, 'pilot_cols')
            p_rows   = double(data.pilot_rows);
            p_cols   = double(data.pilot_cols);
            dmrs_idx = sub2ind([nSubcarriers, nSymbols], p_rows, p_cols);
        elseif isfield(data, 'pilot_indices')
            dmrs_idx = double(data.pilot_indices);
        elseif isfield(data, 'dmrs_idx')
            dmrs_idx = double(data.dmrs_idx);
        else
            dmrs_idx = dmrsIndices; % Use standard 5G NR DMRS configuration indices
        end

        % 5. Extract sparse pilot estimates H_ls_pilots
        if isfield(data, 'H_LS_test')
            H_ls_pilots = double(data.H_LS_test);
        elseif isfield(data, 'H_ls_pilots')
            H_ls_pilots = double(data.H_ls_pilots);
        elseif isfield(data, 'H_ls')
            H_ls_pilots = double(data.H_ls);
        elseif isfield(data, 'H_LS')
            H_ls_pilots = double(data.H_LS);
        else
            H_ls_pilots = [];
        end

        numUE = size(H_perfect_raw, 1);

        % Convert from [numUE x 132 x 14] to cell array of [132 x 14] matrices
        H_perfect_cell = cell(1, numUE);
        H_li_cell      = cell(1, numUE);
        H_infer_cell   = cell(1, numUE);

        for n = 1:numUE
            H_perfect_cell{n} = squeeze(H_perfect_raw(n, :, :));
            H_li_cell{n}      = squeeze(H_li_raw(n, :, :));
            H_infer_cell{n}   = squeeze(H_infer_raw(n, :, :));
        end

        % 6. Check if LMMSE Benchmark calculation is possible
        has_lmmse = ~isempty(dmrs_idx) && ~isempty(H_ls_pilots);
        if has_lmmse
            % If H_ls_pilots is 3D grid, extract pilot elements
            if ndims(H_ls_pilots) == 3
                H_ls_2d = zeros(numUE, length(dmrs_idx));
                for n = 1:numUE
                    grid_tmp = squeeze(H_ls_pilots(n, :, :));
                    H_ls_2d(n, :) = grid_tmp(dmrs_idx);
                end
                H_ls_pilots = H_ls_2d;
            end

            if size(H_ls_pilots, 1) ~= numUE && size(H_ls_pilots, 2) == numUE
                H_ls_pilots = H_ls_pilots.';
            end

            % Build empirical covariance matrix R_hh across all UEs from H_perfect
            nREs = nSubcarriers * nSymbols;
            H_perf_matrix = zeros(nREs, numUE);
            for n = 1:numUE
                h_grid = H_perfect_cell{n};
                H_perf_matrix(:, n) = h_grid(:);
            end

            R_hh = (H_perf_matrix * H_perf_matrix') / numUE;
            R_h_hp = R_hh(:, dmrs_idx);
            R_hp_hp = R_hh(dmrs_idx, dmrs_idx);

            % Compute noise power for MMSE weight matrix
            sigPower = mean(abs(H_ls_pilots(:)).^2);
            snr_linear = 10^(snr_val / 10);
            noisePower = sigPower / snr_linear;

            W_MMSE = R_h_hp / (R_hp_hp + noisePower * eye(length(dmrs_idx)));

            % Formulate H_MMSE for all UEs
            H_mmse_cell = cell(1, numUE);
            for n = 1:numUE
                h_pilot_vec = H_ls_pilots(n, :).';
                h_mmse_vec  = W_MMSE * h_pilot_vec;
                H_mmse_cell{n} = reshape(h_mmse_vec, [nSubcarriers, nSymbols]);
            end
        else
            H_mmse_cell = H_perfect_cell; % Fallback if pilots are unavailable
        end

        % 7. Initialize per-UE metric accumulators
        ber_li_ue    = zeros(1, numUE);
        ber_infer_ue = zeros(1, numUE);
        ber_mmse_ue  = zeros(1, numUE);

        mse_li_ue    = zeros(1, numUE);
        mse_infer_ue = zeros(1, numUE);
        mse_mmse_ue  = zeros(1, numUE);

        nmse_li_ue   = zeros(1, numUE);
        nmse_infer_ue= zeros(1, numUE);
        nmse_mmse_ue = zeros(1, numUE);

        ssim_li_ue   = zeros(1, numUE);
        ssim_infer_ue= zeros(1, numUE);
        ssim_mmse_ue = zeros(1, numUE);

        % Simulation Loop over UEs
        rng(100 + s_idx); % Fixed seed per SNR for reproducible bit sequence
        snr_linear = 10^(snr_val / 10);

        for n = 1:numUE
            H_perf_n  = H_perfect_cell{n};
            H_li_n    = H_li_cell{n};
            H_infer_n = H_infer_cell{n};
            H_mmse_n  = H_mmse_cell{n};

            % Generate random PDSCH payload data bits
            txBits = randi([0 1], pdschInfo.G, 1);
            txSymbols = nrPDSCH(carrier, pdsch, txBits);

            % Build transmit OFDM grid
            txGrid = zeros(nSubcarriers, nSymbols);
            txGrid(pdschIndices) = txSymbols;
            txGrid(dmrsIndices)  = dmrsSymbols;

            % Receive Grid transmission over true channel H_perf_n + AWGN
            rxGrid_clean = txGrid .* H_perf_n;
            sigPwr_grid  = mean(abs(rxGrid_clean(:)).^2);
            N0           = sigPwr_grid / snr_linear;
            noise_grid   = sqrt(N0 / 2) * (randn(size(rxGrid_clean)) + 1j * randn(size(rxGrid_clean)));
            rxGrid       = rxGrid_clean + noise_grid;

            % --- Equalization & Demodulation for H_li ---
            rxData_li = rxGrid(pdschIndices) .* conj(H_li_n(pdschIndices)) ./ (abs(H_li_n(pdschIndices)).^2 + N0);
            llr_li    = nrPDSCHDecode(carrier, pdsch, rxData_li, N0);
            rxBits_li = llr_li{1} < 0;
            ber_li_ue(n) = mean(txBits ~= rxBits_li);

            % --- Equalization & Demodulation for H_infer (Model Output) ---
            rxData_infer = rxGrid(pdschIndices) .* conj(H_infer_n(pdschIndices)) ./ (abs(H_infer_n(pdschIndices)).^2 + N0);
            llr_infer    = nrPDSCHDecode(carrier, pdsch, rxData_infer, N0);
            rxBits_infer = llr_infer{1} < 0;
            ber_infer_ue(n) = mean(txBits ~= rxBits_infer);

            % --- Equalization & Demodulation for H_MMSE ---
            rxData_mmse = rxGrid(pdschIndices) .* conj(H_mmse_n(pdschIndices)) ./ (abs(H_mmse_n(pdschIndices)).^2 + N0);
            llr_mmse    = nrPDSCHDecode(carrier, pdsch, rxData_mmse, N0);
            rxBits_mmse = llr_mmse{1} < 0;
            ber_mmse_ue(n) = mean(txBits ~= rxBits_mmse);

            % --- Channel Estimation Metrics: MSE, NMSE, SSIM ---
            % MSE
            mse_li_ue(n)    = mean(abs(H_li_n(:) - H_perf_n(:)).^2);
            mse_infer_ue(n) = mean(abs(H_infer_n(:) - H_perf_n(:)).^2);
            mse_mmse_ue(n)  = mean(abs(H_mmse_n(:) - H_perf_n(:)).^2);

            % NMSE
            norm_gt = sum(abs(H_perf_n(:)).^2);
            nmse_li_ue(n)    = sum(abs(H_li_n(:) - H_perf_n(:)).^2) / norm_gt;
            nmse_infer_ue(n) = sum(abs(H_infer_n(:) - H_perf_n(:)).^2) / norm_gt;
            nmse_mmse_ue(n)  = sum(abs(H_mmse_n(:) - H_perf_n(:)).^2) / norm_gt;

            % SSIM (Normalized magnitude SSIM)
            max_val = max(abs(H_perf_n(:)));
            if max_val > 0
                img_gt    = abs(H_perf_n) / max_val;
                img_li    = abs(H_li_n) / max_val;
                img_infer = abs(H_infer_n) / max_val;
                img_mmse  = abs(H_mmse_n) / max_val;

                ssim_li_ue(n)    = compute_ssim_robust(img_li, img_gt);
                ssim_infer_ue(n) = compute_ssim_robust(img_infer, img_gt);
                ssim_mmse_ue(n)  = compute_ssim_robust(img_mmse, img_gt);
            end
        end

        % Store average metrics for current SNR
        ber_li_arr(s_idx)    = mean(ber_li_ue);
        ber_infer_arr(s_idx) = mean(ber_infer_ue);
        ber_mmse_arr(s_idx)  = mean(ber_mmse_ue);

        mse_li_arr(s_idx)    = mean(mse_li_ue);
        mse_infer_arr(s_idx) = mean(mse_infer_ue);
        mse_mmse_arr(s_idx)  = mean(mse_mmse_ue);

        nmse_li_arr(s_idx)    = mean(nmse_li_ue);
        nmse_infer_arr(s_idx) = mean(nmse_infer_ue);
        nmse_mmse_arr(s_idx)  = mean(nmse_mmse_ue);

        nmse_li_db_arr(s_idx)    = 10 * log10(mean(nmse_li_ue));
        nmse_infer_db_arr(s_idx) = 10 * log10(mean(nmse_infer_ue));
        nmse_mmse_db_arr(s_idx)  = 10 * log10(mean(nmse_mmse_ue));

        ssim_li_arr(s_idx)    = mean(ssim_li_ue);
        ssim_infer_arr(s_idx) = mean(ssim_infer_ue);
        ssim_mmse_arr(s_idx)  = mean(ssim_mmse_ue);

        % Save BER & Channel metrics inside current SNR subfolder
        eval_mat_file = fullfile(sf_path, sprintf('BER_performance_results_%s.mat', domain));
        save(eval_mat_file, ...
            'snr_val', 'ber_li_ue', 'ber_infer_ue', 'ber_mmse_ue', ...
            'mse_li_ue', 'mse_infer_ue', 'mse_mmse_ue', ...
            'nmse_li_ue', 'nmse_infer_ue', 'nmse_mmse_ue', ...
            'ssim_li_ue', 'ssim_infer_ue', 'ssim_mmse_ue');
        
        % For target domain, also save standard BER_performance_results.mat
        if strcmp(domain, 'target')
            save(fullfile(sf_path, 'BER_performance_results.mat'), ...
                'snr_val', 'ber_li_ue', 'ber_infer_ue', 'ber_mmse_ue', ...
                'mse_li_ue', 'mse_infer_ue', 'mse_mmse_ue', ...
                'nmse_li_ue', 'nmse_infer_ue', 'nmse_mmse_ue', ...
                'ssim_li_ue', 'ssim_infer_ue', 'ssim_mmse_ue');
        end
    end

    % Print console summary table
    fprintf('\n========================================================================\n');
    fprintf('           PERFORMANCE SUMMARY - %s DOMAIN\n', upper(domain));
    fprintf('========================================================================\n');
    fprintf('%-8s | %-12s %-12s %-12s | %-12s %-12s %-12s\n', ...
        'SNR (dB)', 'BER (LI)', 'BER (Infer)', 'BER (MMSE)', 'NMSEdB (LI)', 'NMSEdB(Inf)', 'NMSEdB(MMSE)');
    fprintf('%s\n', repmat('-', 1, 95));
    for i = 1:num_snr
        fprintf('%-8.1f | %-12.6f %-12.6f %-12.6f | %-12.2f %-12.2f %-12.2f\n', ...
            SNRdB_sorted(i), ber_li_arr(i), ber_infer_arr(i), ber_mmse_arr(i), ...
            nmse_li_db_arr(i), nmse_infer_db_arr(i), nmse_mmse_db_arr(i));
    end
    fprintf('\n');

    fprintf('%-8s | %-12s %-12s %-12s | %-12s %-12s %-12s\n', ...
        'SNR (dB)', 'SSIM (LI)', 'SSIM (Infer)', 'SSIM (MMSE)', 'MSE (LI)', 'MSE (Infer)', 'MSE (MMSE)');
    fprintf('%s\n', repmat('-', 1, 95));
    for i = 1:num_snr
        fprintf('%-8.1f | %-12.4f %-12.4f %-12.4f | %-12.3e %-12.3e %-12.3e\n', ...
            SNRdB_sorted(i), ssim_li_arr(i), ssim_infer_arr(i), ssim_mmse_arr(i), ...
            mse_li_arr(i), mse_infer_arr(i), mse_mmse_arr(i));
    end
    fprintf('\n');

    % Compose output struct
    save_struct = struct();
    save_struct.domain          = domain;
    save_struct.SNRdB           = SNRdB_sorted;
    save_struct.ber_li          = ber_li_arr;
    save_struct.ber_infer       = ber_infer_arr;
    save_struct.ber_mmse        = ber_mmse_arr;

    save_struct.mse_li          = mse_li_arr;
    save_struct.mse_infer       = mse_infer_arr;
    save_struct.mse_mmse        = mse_mmse_arr;

    save_struct.nmse_li         = nmse_li_arr;
    save_struct.nmse_infer      = nmse_infer_arr;
    save_struct.nmse_mmse       = nmse_mmse_arr;

    save_struct.nmse_li_dB      = nmse_li_db_arr;
    save_struct.nmse_infer_dB   = nmse_infer_db_arr;
    save_struct.nmse_mmse_dB    = nmse_mmse_db_arr;

    save_struct.ssim_li         = ssim_li_arr;
    save_struct.ssim_infer      = ssim_infer_arr;
    save_struct.ssim_mmse       = ssim_mmse_arr;

    % Save domain-specific MAT file
    domain_mat_path = fullfile(batch_folder, sprintf('synthesized_results_%s.mat', domain));
    save(domain_mat_path, '-struct', 'save_struct');
    fprintf('[Saved] Synthesized MAT file (%s) -> %s\n', domain, domain_mat_path);

    % Plot & Export PDF Figures for this Domain
    plot_domain_figures(batch_folder, SNRdB_sorted, save_struct, domain, labelname);

    % Export domain-specific markdown report
    md_path = fullfile(batch_folder, sprintf('simulation_results_%s.md', domain));
    export_markdown_report(md_path, batch_folder, SNRdB_sorted, save_struct, domain, labelname);
end

%% Plotting & exporting figures for a specific domain
function plot_domain_figures(batch_folder, SNRdB_sorted, s, domain, labelname)
    domain_label = sprintf('%s (%s)', labelname, [upper(domain(1)), domain(2:end)]);
    
    % Colors
    col_li    = [0 0.4470 0.7410];
    col_infer = [0.8500 0.3250 0.0980];
    col_mmse  = [0 0 0];

    % 1. BER Comparison
    fig1 = figure('Name', sprintf('BER Comparison (%s)', domain), 'Color', 'w', 'Position', [100 100 700 500], 'Visible', 'off');
    hold on;
    h1 = semilogy(SNRdB_sorted, s.ber_li, '--o', 'Color', col_li, 'LineWidth', 1.8, 'MarkerSize', 7, 'MarkerFaceColor', col_li);
    h2 = semilogy(SNRdB_sorted, s.ber_infer, '-^', 'Color', col_infer, 'LineWidth', 2.0, 'MarkerSize', 8, 'MarkerFaceColor', col_infer);
    h3 = semilogy(SNRdB_sorted, s.ber_mmse, '-s', 'Color', col_mmse, 'LineWidth', 1.8, 'MarkerSize', 7, 'MarkerFaceColor', col_mmse);
    xlabel('SNR (dB)', 'FontSize', 12, 'FontWeight', 'bold');
    ylabel('Bit Error Rate (BER)', 'FontSize', 12, 'FontWeight', 'bold');
    title(sprintf('BER Performance (%s Domain)', upper(domain)), 'FontSize', 14, 'FontWeight', 'bold');
    grid on; box on; set(gca, 'YMinorGrid', 'on', 'FontSize', 11);
    legend([h1, h2, h3], {'LS + Linear Interpolation', domain_label, 'MMSE Benchmark'}, 'Location', 'southwest', 'FontSize', 10);
    hold off;
    
    save_fig(fig1, fullfile(batch_folder, sprintf('BER_comparison_%s.pdf', domain)));
    if strcmp(domain, 'target')
        save_fig(fig1, fullfile(batch_folder, 'BER_comparison.pdf'));
    end
    close(fig1);

    % 2. NMSE Comparison (dB)
    fig2 = figure('Name', sprintf('NMSE Comparison (%s)', domain), 'Color', 'w', 'Position', [150 150 700 500], 'Visible', 'off');
    hold on;
    h1 = plot(SNRdB_sorted, s.nmse_li_dB, '--o', 'Color', col_li, 'LineWidth', 1.8, 'MarkerSize', 7, 'MarkerFaceColor', col_li);
    h2 = plot(SNRdB_sorted, s.nmse_infer_dB, '-^', 'Color', col_infer, 'LineWidth', 2.0, 'MarkerSize', 8, 'MarkerFaceColor', col_infer);
    h3 = plot(SNRdB_sorted, s.nmse_mmse_dB, '-s', 'Color', col_mmse, 'LineWidth', 1.8, 'MarkerSize', 7, 'MarkerFaceColor', col_mmse);
    xlabel('SNR (dB)', 'FontSize', 12, 'FontWeight', 'bold');
    ylabel('Normalized Mean Squared Error (NMSE) [dB]', 'FontSize', 12, 'FontWeight', 'bold');
    title(sprintf('NMSE Performance (%s Domain)', upper(domain)), 'FontSize', 14, 'FontWeight', 'bold');
    grid on; box on; set(gca, 'YMinorGrid', 'on', 'FontSize', 11);
    legend([h1, h2, h3], {'LS + Linear Interpolation', domain_label, 'MMSE Benchmark'}, 'Location', 'northeast', 'FontSize', 10);
    hold off;

    save_fig(fig2, fullfile(batch_folder, sprintf('NMSE_comparison_%s.pdf', domain)));
    if strcmp(domain, 'target')
        save_fig(fig2, fullfile(batch_folder, 'NMSE_comparison.pdf'));
    end
    close(fig2);

    % 3. SSIM Comparison
    fig3 = figure('Name', sprintf('SSIM Comparison (%s)', domain), 'Color', 'w', 'Position', [200 200 700 500], 'Visible', 'off');
    hold on;
    h1 = plot(SNRdB_sorted, s.ssim_li, '--o', 'Color', col_li, 'LineWidth', 1.8, 'MarkerSize', 7, 'MarkerFaceColor', col_li);
    h2 = plot(SNRdB_sorted, s.ssim_infer, '-^', 'Color', col_infer, 'LineWidth', 2.0, 'MarkerSize', 8, 'MarkerFaceColor', col_infer);
    h3 = plot(SNRdB_sorted, s.ssim_mmse, '-s', 'Color', col_mmse, 'LineWidth', 1.8, 'MarkerSize', 7, 'MarkerFaceColor', col_mmse);
    xlabel('SNR (dB)', 'FontSize', 12, 'FontWeight', 'bold');
    ylabel('Structural Similarity Index (SSIM)', 'FontSize', 12, 'FontWeight', 'bold');
    title(sprintf('SSIM Performance (%s Domain)', upper(domain)), 'FontSize', 14, 'FontWeight', 'bold');
    grid on; box on; ylim([0 1.05]); set(gca, 'YMinorGrid', 'on', 'FontSize', 11);
    legend([h1, h2, h3], {'LS + Linear Interpolation', domain_label, 'MMSE Benchmark'}, 'Location', 'southeast', 'FontSize', 10);
    hold off;

    save_fig(fig3, fullfile(batch_folder, sprintf('SSIM_comparison_%s.pdf', domain)));
    if strcmp(domain, 'target')
        save_fig(fig3, fullfile(batch_folder, 'SSIM_comparison.pdf'));
    end
    close(fig3);

    % 4. MSE Comparison
    fig4 = figure('Name', sprintf('MSE Comparison (%s)', domain), 'Color', 'w', 'Position', [250 250 700 500], 'Visible', 'off');
    hold on;
    h1 = semilogy(SNRdB_sorted, s.mse_li, '--o', 'Color', col_li, 'LineWidth', 1.8, 'MarkerSize', 7, 'MarkerFaceColor', col_li);
    h2 = semilogy(SNRdB_sorted, s.mse_infer, '-^', 'Color', col_infer, 'LineWidth', 2.0, 'MarkerSize', 8, 'MarkerFaceColor', col_infer);
    h3 = semilogy(SNRdB_sorted, s.mse_mmse, '-s', 'Color', col_mmse, 'LineWidth', 1.8, 'MarkerSize', 7, 'MarkerFaceColor', col_mmse);
    xlabel('SNR (dB)', 'FontSize', 12, 'FontWeight', 'bold');
    ylabel('Mean Squared Error (MSE)', 'FontSize', 12, 'FontWeight', 'bold');
    title(sprintf('MSE Performance (%s Domain)', upper(domain)), 'FontSize', 14, 'FontWeight', 'bold');
    grid on; box on; set(gca, 'YMinorGrid', 'on', 'FontSize', 11);
    legend([h1, h2, h3], {'LS + Linear Interpolation', domain_label, 'MMSE Benchmark'}, 'Location', 'northeast', 'FontSize', 10);
    hold off;

    save_fig(fig4, fullfile(batch_folder, sprintf('MSE_comparison_%s.pdf', domain)));
    if strcmp(domain, 'target')
        save_fig(fig4, fullfile(batch_folder, 'MSE_comparison.pdf'));
    end
    close(fig4);
end

%% Plot side-by-side Source vs Target comparison curves
function plot_source_vs_target_figures(batch_folder, SNRdB_sorted, src, tgt, labelname)
    % 1. BER Comparison: Source vs Target
    fig_ber = figure('Name', 'BER Source vs Target', 'Color', 'w', 'Position', [100 100 750 520], 'Visible', 'off');
    hold on;
    h1 = semilogy(SNRdB_sorted, src.ber_infer, '-o', 'Color', [0.2 0.6 0.2], 'LineWidth', 2.0, 'MarkerSize', 7, 'MarkerFaceColor', [0.2 0.6 0.2]);
    h2 = semilogy(SNRdB_sorted, tgt.ber_infer, '-^', 'Color', [0.8500 0.3250 0.0980], 'LineWidth', 2.0, 'MarkerSize', 8, 'MarkerFaceColor', [0.8500 0.3250 0.0980]);
    h3 = semilogy(SNRdB_sorted, tgt.ber_li, '--d', 'Color', [0 0.4470 0.7410], 'LineWidth', 1.6, 'MarkerSize', 6);
    h4 = semilogy(SNRdB_sorted, tgt.ber_mmse, '-s', 'Color', [0 0 0], 'LineWidth', 1.8, 'MarkerSize', 7, 'MarkerFaceColor', [0 0 0]);
    xlabel('SNR (dB)', 'FontSize', 12, 'FontWeight', 'bold');
    ylabel('Bit Error Rate (BER)', 'FontSize', 12, 'FontWeight', 'bold');
    title(sprintf('BER: Source vs Target Domain Adaptation (%s)', labelname), 'FontSize', 13, 'FontWeight', 'bold');
    grid on; box on; set(gca, 'YMinorGrid', 'on', 'FontSize', 11);
    legend([h1, h2, h3, h4], ...
        {sprintf('%s (Source)', labelname), sprintf('%s (Target)', labelname), 'LS + LI Benchmark (Target)', 'MMSE Benchmark (Target)'}, ...
        'Location', 'southwest', 'FontSize', 10);
    hold off;
    save_fig(fig_ber, fullfile(batch_folder, 'BER_comparison_source_vs_target.pdf'));
    close(fig_ber);

    % 2. NMSE (dB) Comparison: Source vs Target
    fig_nmse = figure('Name', 'NMSE Source vs Target', 'Color', 'w', 'Position', [150 150 750 520], 'Visible', 'off');
    hold on;
    h1 = plot(SNRdB_sorted, src.nmse_infer_dB, '-o', 'Color', [0.2 0.6 0.2], 'LineWidth', 2.0, 'MarkerSize', 7, 'MarkerFaceColor', [0.2 0.6 0.2]);
    h2 = plot(SNRdB_sorted, tgt.nmse_infer_dB, '-^', 'Color', [0.8500 0.3250 0.0980], 'LineWidth', 2.0, 'MarkerSize', 8, 'MarkerFaceColor', [0.8500 0.3250 0.0980]);
    h3 = plot(SNRdB_sorted, tgt.nmse_li_dB, '--d', 'Color', [0 0.4470 0.7410], 'LineWidth', 1.6, 'MarkerSize', 6);
    h4 = plot(SNRdB_sorted, tgt.nmse_mmse_dB, '-s', 'Color', [0 0 0], 'LineWidth', 1.8, 'MarkerSize', 7, 'MarkerFaceColor', [0 0 0]);
    xlabel('SNR (dB)', 'FontSize', 12, 'FontWeight', 'bold');
    ylabel('Normalized Mean Squared Error (NMSE) [dB]', 'FontSize', 12, 'FontWeight', 'bold');
    title(sprintf('NMSE (dB): Source vs Target Domain Adaptation (%s)', labelname), 'FontSize', 13, 'FontWeight', 'bold');
    grid on; box on; set(gca, 'YMinorGrid', 'on', 'FontSize', 11);
    legend([h1, h2, h3, h4], ...
        {sprintf('%s (Source)', labelname), sprintf('%s (Target)', labelname), 'LS + LI Benchmark (Target)', 'MMSE Benchmark (Target)'}, ...
        'Location', 'northeast', 'FontSize', 10);
    hold off;
    save_fig(fig_nmse, fullfile(batch_folder, 'NMSE_comparison_source_vs_target.pdf'));
    close(fig_nmse);
end

%% Helper to safely save figure to PDF
function save_fig(fig_handle, pdf_path)
    try
        exportgraphics(fig_handle, pdf_path, 'ContentType', 'vector');
    catch
        try
            saveas(fig_handle, pdf_path);
        catch err
            warning('Could not export PDF to %s: %s', pdf_path, err.message);
        end
    end
    fprintf('[Saved Figure] %s\n', pdf_path);
end

%% Markdown export helpers
function export_markdown_report(md_path, batch_folder, SNRdB_sorted, s, domain, labelname)
    fid = fopen(md_path, 'w');
    if fid == -1
        return;
    end
    fprintf(fid, '# CORAL UDA %s Domain Performance Summary\n\n', upper(domain));
    fprintf(fid, '**Batch Directory:** `%s`  \n', batch_folder);
    fprintf(fid, '**Model Label:** `%s`  \n\n', labelname);

    fprintf(fid, '## BER Performance Table\n');
    fprintf(fid, '| SNR (dB) | LS + Linear Interpolation | %s | MMSE Benchmark |\n', labelname);
    fprintf(fid, '|:---:|:---:|:---:|:---:|\n');
    for i = 1:length(SNRdB_sorted)
        fprintf(fid, '| %.1f | %.6f | %.6f | %.6f |\n', SNRdB_sorted(i), s.ber_li(i), s.ber_infer(i), s.ber_mmse(i));
    end
    fprintf(fid, '\n');

    fprintf(fid, '## NMSE (dB) Performance Table\n');
    fprintf(fid, '| SNR (dB) | LS + Linear Interpolation | %s | MMSE Benchmark |\n', labelname);
    fprintf(fid, '|:---:|:---:|:---:|:---:|\n');
    for i = 1:length(SNRdB_sorted)
        fprintf(fid, '| %.1f | %.2f dB | %.2f dB | %.2f dB |\n', SNRdB_sorted(i), s.nmse_li_dB(i), s.nmse_infer_dB(i), s.nmse_mmse_dB(i));
    end
    fprintf(fid, '\n');

    fprintf(fid, '## SSIM Performance Table\n');
    fprintf(fid, '| SNR (dB) | LS + Linear Interpolation | %s | MMSE Benchmark |\n', labelname);
    fprintf(fid, '|:---:|:---:|:---:|:---:|\n');
    for i = 1:length(SNRdB_sorted)
        fprintf(fid, '| %.1f | %.4f | %.4f | %.4f |\n', SNRdB_sorted(i), s.ssim_li(i), s.ssim_infer(i), s.ssim_mmse(i));
    end
    fclose(fid);
    fprintf('[Saved Report] %s\n', md_path);
end

function generate_unified_markdown_report(batch_folder, SNRdB_sorted, domain_results, labelname)
    md_path = fullfile(batch_folder, 'simulation_results.md');
    fid = fopen(md_path, 'w');
    if fid == -1
        return;
    end
    fprintf(fid, '# CORAL UDA Channel Equalization & Domain Performance Summary\n\n');
    fprintf(fid, '**Batch Directory:** `%s`  \n', batch_folder);
    fprintf(fid, '**Model Evaluation:** `%s`  \n\n', labelname);

    if isfield(domain_results, 'target') && isfield(domain_results, 'source')
        src = domain_results.source;
        tgt = domain_results.target;

        fprintf(fid, '## Multi-Domain BER Comparison Table\n');
        fprintf(fid, '| SNR (dB) | Source Inferred | Target Inferred | Target LI Benchmark | Target MMSE Benchmark |\n');
        fprintf(fid, '|:---:|:---:|:---:|:---:|:---:|\n');
        for i = 1:length(SNRdB_sorted)
            fprintf(fid, '| %.1f | %.6f | %.6f | %.6f | %.6f |\n', ...
                SNRdB_sorted(i), src.ber_infer(i), tgt.ber_infer(i), tgt.ber_li(i), tgt.ber_mmse(i));
        end
        fprintf(fid, '\n');

        fprintf(fid, '## Multi-Domain NMSE (dB) Comparison Table\n');
        fprintf(fid, '| SNR (dB) | Source Inferred | Target Inferred | Target LI Benchmark | Target MMSE Benchmark |\n');
        fprintf(fid, '|:---:|:---:|:---:|:---:|:---:|\n');
        for i = 1:length(SNRdB_sorted)
            fprintf(fid, '| %.1f | %.2f dB | %.2f dB | %.2f dB | %.2f dB |\n', ...
                SNRdB_sorted(i), src.nmse_infer_dB(i), tgt.nmse_infer_dB(i), tgt.nmse_li_dB(i), tgt.nmse_mmse_dB(i));
        end
        fprintf(fid, '\n');

        fprintf(fid, '## Multi-Domain SSIM Comparison Table\n');
        fprintf(fid, '| SNR (dB) | Source Inferred | Target Inferred | Target LI Benchmark | Target MMSE Benchmark |\n');
        fprintf(fid, '|:---:|:---:|:---:|:---:|:---:|\n');
        for i = 1:length(SNRdB_sorted)
            fprintf(fid, '| %.1f | %.4f | %.4f | %.4f | %.4f |\n', ...
                SNRdB_sorted(i), src.ssim_infer(i), tgt.ssim_infer(i), tgt.ssim_li(i), tgt.ssim_mmse(i));
        end
    end
    fclose(fid);
    fprintf('[Saved Unified Report] %s\n', md_path);
end

%% Robust SSIM calculation
function val = compute_ssim_robust(x, y)
    try
        val = ssim(x, y);
    catch
        % Fallback SSIM calculation without Image Processing Toolbox
        K1 = 0.01; K2 = 0.03; L = 1.0;
        C1 = (K1 * L)^2; C2 = (K2 * L)^2;
        mu_x = mean(x(:)); mu_y = mean(y(:));
        var_x = var(x(:)); var_y = var(y(:));
        cov_xy = mean((x(:) - mu_x) .* (y(:) - mu_y));
        
        num = (2 * mu_x * mu_y + C1) * (2 * cov_xy + C2);
        den = (mu_x^2 + mu_y^2 + C1) * (var_x + var_y + C2);
        val = num / den;
    end
end

%% Align channel matrix dimensions to [N x nSubcarriers x nSymbols]
function arr = align_channel_matrix(arr, nSubc, nSymb)
    if isempty(arr) || ndims(arr) ~= 3
        return;
    end
    sz = size(arr);
    % If already [N x 132 x 14]
    if sz(2) == nSubc && sz(3) == nSymb
        return;
    end
    % If [N x 14 x 132] -> permute to [N x 132 x 14]
    if sz(2) == nSymb && sz(3) == nSubc
        arr = permute(arr, [1, 3, 2]);
        return;
    end
    % If [14 x 132 x N] -> permute to [N x 132 x 14]
    if sz(1) == nSymb && sz(2) == nSubc
        arr = permute(arr, [3, 2, 1]);
        return;
    end
    % If [132 x 14 x N] -> permute to [N x 132 x 14]
    if sz(1) == nSubc && sz(2) == nSymb
        arr = permute(arr, [3, 1, 2]);
        return;
    end
end
