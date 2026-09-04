% SYN_SYN_COMPARE_MULTIMODELS Load synthesized_results from multiple model/layer folders
% and plot multi-model performance comparisons (MSE, NMSE, SSIM, BER) for both
% Source and Target domains separately.
%
% Synthesizes and plots comparison curves for multiple inferred models alongside
% averaged benchmark curves (LI Benchmark Average and MMSE Benchmark Average).
%
% Usage:
%   syn_syn_compare_multiModels()
%   syn_syn_compare_multiModels(folders)
%   syn_syn_compare_multiModels(folders, folder_labels)
%   syn_syn_compare_multiModels(folders, folder_labels, output_folder)
%
% Inputs:
%   folders       - Cell array of folder paths containing synthesized_results_*.mat
%   folder_labels - Cell array of custom legend names for each folder's model
%                   (e.g., {'LS+Transformer CORAL layer 1', 'LS+Transformer CORAL layer 1,2'})
%   output_folder - (Optional) Directory to save export PDFs and comparison MAT files.
%
% Output:
%   results       - Struct with synthesized comparison data for both Source and Target domains.
%
function results = syn_syn_compare_multiModels(folders, folder_labels, output_folder)
    script_dir = fileparts(mfilename('fullpath'));
    if ~isempty(script_dir) && exist(script_dir, 'dir')
        cd(script_dir);
    end
    
    folders_ = { ...
            'C:\Users\AT30890\Hoctap\1_Hprediction\working\H_predict_NTN\Hest_NTN_UDA_clean\CORAL\A100__DUR100_2p18e9_600km_30kHz\LS_Attention_standardize\layer1', ...
            'C:\Users\AT30890\Hoctap\1_Hprediction\working\H_predict_NTN\Hest_NTN_UDA_clean\CORAL\A100__DUR100_2p18e9_600km_30kHz\LS_Attention_standardize\layer1_layer2' ...
        };
    folder_labels_ = {'LS+Transformer CORAL layer 1', 'LS+Transformer CORAL layer 1,2'};

    % Default Folders Fallback
    if nargin < 1 || isempty(folders)
        folders = folders_;
    end

    if ischar(folders) || isstring(folders)
        folders = {char(folders)};
    end

    num_folders = length(folders);

    % Default Folder Labels Fallback
    if nargin < 2 || isempty(folder_labels)
        folder_labels = folder_labels_;
    elseif ischar(folder_labels) || isstring(folder_labels)
        folder_labels = {char(folder_labels)};
    end

    % Fill missing labels if necessary
    for i = (length(folder_labels) + 1):num_folders
        [~, leaf_name] = fileparts(folders{i});
        folder_labels{i} = sprintf('Model %d (%s)', i, strrep(leaf_name, '_', ' '));
    end

    % Default Output Directory (Incremental syn_x in parent directory if not specified)
    if nargin < 3 || isempty(output_folder)
        parent_dir = fileparts(folders{1});
        max_syn = 0;
        if exist(parent_dir, 'dir')
            sub_items = dir(parent_dir);
            for k = 1:length(sub_items)
                if sub_items(k).isdir
                    tok = regexp(sub_items(k).name, '^syn_(\d+)$', 'tokens');
                    if ~isempty(tok)
                        num = str2double(tok{1}{1});
                        if num > max_syn
                            max_syn = num;
                        end
                    end
                end
            end
        end
        output_folder = fullfile(parent_dir, sprintf('syn_%d', max_syn + 1));
    end

    if ~exist(output_folder, 'dir')
        mkdir(output_folder);
    end

    safe_printf('========================================================================\n');
    safe_printf('           CORAL Multi-Model Synthesized Results Comparison             \n');
    safe_printf('========================================================================\n');
    safe_printf('Output Directory: %s\n\n', output_folder);
    safe_printf('Loading %d Synthesized Results Folders:\n', num_folders);
    for i = 1:num_folders
        safe_printf('  [%d] Label: "%s"\n      Path:  %s\n', i, folder_labels{i}, folders{i});
    end
    safe_printf('\n');

    % Record plotted folders and labels configuration
    fid_txt = fopen(fullfile(output_folder, 'plotted_folders.txt'), 'w');
    if fid_txt ~= -1
        fprintf(fid_txt, "Plotted Folders & Labels Configuration:\n");
        for i = 1:num_folders
            fprintf(fid_txt, "[%d] %s -> %s\n", i, folder_labels{i}, folders{i});
        end
        fclose(fid_txt);
    end

    % Domains to compare separately: Target Domain (primary) and Source Domain
    domains = {'target', 'source'};
    comparison_results = struct();

    for d_idx = 1:length(domains)
        domain = domains{d_idx};
        safe_printf('------------------------------------------------------------------------\n');
        safe_printf('>>> Comparing Multi-Model Performance: %s DOMAIN <<<\n', upper(domain));
        safe_printf('------------------------------------------------------------------------\n');

        dom_res = compare_domain_multi_models(folders, folder_labels, output_folder, domain);
        comparison_results.(domain) = dom_res;
    end

    % Compose unified output struct
    results = struct();
    if isfield(comparison_results, 'target') && ~isempty(comparison_results.target)
        results = comparison_results.target;
    end
    results.target = comparison_results.target;
    results.source = comparison_results.source;

    % Save unified comparison MAT file
    unified_mat_path = fullfile(output_folder, 'synthesized_comparison_results.mat');
    save(unified_mat_path, '-struct', 'results');
    safe_printf('\n[Saved] Unified comparative MAT file -> %s\n', unified_mat_path);

    % Generate unified markdown comparison report
    generate_unified_markdown_comparison_report(output_folder, comparison_results, folder_labels);

    safe_printf('\n========================================================================\n');
    safe_printf(' Multi-Model Synthesized Comparison Completed Successfully!\n');
    safe_printf(' Destination: %s\n', output_folder);
    safe_printf('========================================================================\n');
end

%% Domain-specific multi-model comparison function
function domain_struct = compare_domain_multi_models(folders, folder_labels, output_folder, domain)
    num_folders = length(folders);
    models_data = cell(1, num_folders);
    SNRdB = [];

    % Track extracted metrics per folder
    m_mse     = cell(1, num_folders);
    m_nmse    = cell(1, num_folders);
    m_nmse_db = cell(1, num_folders);
    m_ssim    = cell(1, num_folders);
    m_ber     = cell(1, num_folders);

    li_mse     = cell(1, num_folders);
    li_nmse    = cell(1, num_folders);
    li_nmse_db = cell(1, num_folders);
    li_ssim    = cell(1, num_folders);
    li_ber     = cell(1, num_folders);

    mmse_mse     = cell(1, num_folders);
    mmse_nmse    = cell(1, num_folders);
    mmse_nmse_db = cell(1, num_folders);
    mmse_ssim    = cell(1, num_folders);
    mmse_ber     = cell(1, num_folders);

    mat_filename = sprintf('synthesized_results_%s.mat', domain);

    % Load and Parse MAT files
    for i = 1:num_folders
        mat_path = fullfile(folders{i}, mat_filename);
        if ~exist(mat_path, 'file')
            % Fallback check: synthesized_results.mat
            alt_path = fullfile(folders{i}, 'synthesized_results.mat');
            if exist(alt_path, 'file')
                mat_path = alt_path;
            else
                error('File not found: %s\nPlease run syn_metrics_withBER() on folder:\n  %s', mat_path, folders{i});
            end
        end

        raw_data = load(mat_path);
        
        % Check if domain-specific substruct exists inside synthesized_results.mat
        if isfield(raw_data, domain) && isstruct(raw_data.(domain))
            data = raw_data.(domain);
        else
            data = raw_data;
        end
        models_data{i} = data;

        % Retrieve SNR vector
        current_snr = [];
        if isfield(data, 'SNRdB')
            current_snr = double(data.SNRdB(:).');
        elseif isfield(data, 'snr')
            current_snr = double(data.snr(:).');
        end

        if isempty(SNRdB)
            SNRdB = current_snr;
        elseif ~isempty(current_snr) && ~isequal(SNRdB, current_snr)
            warning('SNRdB mismatch in %s for %s! Aligning with first folder SNRdB.', domain, folders{i});
        end

        num_snr = length(SNRdB);

        % Extract Inferred Model Metrics
        m_mse{i}     = extract_field(data, {'mse_infer', 'mse_test', 'mse_output'}, 'mse_', {'li', 'mmse', 'benchmark', 'input'}, num_snr);
        m_nmse{i}    = extract_field(data, {'nmse_infer', 'nmse_test', 'nmse_output'}, 'nmse_', {'li', 'mmse', 'benchmark', 'input', 'db'}, num_snr);
        m_nmse_db{i} = extract_field(data, {'nmse_infer_dB', 'nmse_infer_db', 'nmse_test_db', 'nmse_test_dB'}, 'nmse_', {'li', 'mmse', 'benchmark', 'input'}, num_snr, true);
        m_ssim{i}    = extract_field(data, {'ssim_infer', 'ssim_test', 'ssim_output'}, 'ssim_', {'li', 'mmse', 'benchmark', 'input'}, num_snr);
        m_ber{i}     = extract_field(data, {'ber_infer', 'ber_test', 'ber_output'}, 'ber_', {'li', 'mmse'}, num_snr);

        % Fill NMSE_dB if linear NMSE exists
        if all(isnan(m_nmse_db{i})) && ~all(isnan(m_nmse{i}))
            m_nmse_db{i} = 10 * log10(m_nmse{i});
        elseif all(isnan(m_nmse{i})) && ~all(isnan(m_nmse_db{i}))
            m_nmse{i} = 10.^(m_nmse_db{i} / 10);
        end

        % Extract LI Benchmark Metrics
        li_mse{i}     = extract_field(data, {'mse_li', 'mse_LI'}, 'mse_li', {}, num_snr);
        li_nmse{i}    = extract_field(data, {'nmse_li', 'nmse_LI'}, 'nmse_li', {'db'}, num_snr);
        li_nmse_db{i} = extract_field(data, {'nmse_li_dB', 'nmse_li_db', 'nmse_LI_dB'}, 'nmse_li', {}, num_snr, true);
        li_ssim{i}    = extract_field(data, {'ssim_li', 'ssim_LI'}, 'ssim_li', {}, num_snr);
        li_ber{i}     = extract_field(data, {'ber_li', 'ber_LI'}, 'ber_li', {}, num_snr);

        if all(isnan(li_nmse_db{i})) && ~all(isnan(li_nmse{i}))
            li_nmse_db{i} = 10 * log10(li_nmse{i});
        end

        % Extract MMSE Benchmark Metrics
        mmse_mse{i}     = extract_field(data, {'mse_mmse', 'mse_MMSE'}, 'mse_mmse', {}, num_snr);
        mmse_nmse{i}    = extract_field(data, {'nmse_mmse', 'nmse_MMSE'}, 'nmse_mmse', {'db'}, num_snr);
        mmse_nmse_db{i} = extract_field(data, {'nmse_mmse_dB', 'nmse_mmse_db', 'nmse_MMSE_dB'}, 'nmse_mmse', {}, num_snr, true);
        mmse_ssim{i}    = extract_field(data, {'ssim_mmse', 'ssim_MMSE'}, 'ssim_mmse', {}, num_snr);
        mmse_ber{i}     = extract_field(data, {'ber_mmse', 'ber_MMSE'}, 'ber_mmse', {}, num_snr);

        if all(isnan(mmse_nmse_db{i})) && ~all(isnan(mmse_nmse{i}))
            mmse_nmse_db{i} = 10 * log10(mmse_nmse{i});
        end
    end

    num_snr = length(SNRdB);

    % Compute Average Benchmarks Across All Folders
    li_mse_avg     = compute_benchmark_avg(li_mse, num_snr);
    li_nmse_avg    = compute_benchmark_avg(li_nmse, num_snr);
    li_nmse_db_avg = compute_benchmark_avg(li_nmse_db, num_snr);
    li_ssim_avg    = compute_benchmark_avg(li_ssim, num_snr);
    li_ber_avg     = compute_benchmark_avg(li_ber, num_snr);

    mmse_mse_avg     = compute_benchmark_avg(mmse_mse, num_snr);
    mmse_nmse_avg    = compute_benchmark_avg(mmse_nmse, num_snr);
    mmse_nmse_db_avg = compute_benchmark_avg(mmse_nmse_db, num_snr);
    mmse_ssim_avg    = compute_benchmark_avg(mmse_ssim, num_snr);
    mmse_ber_avg     = compute_benchmark_avg(mmse_ber, num_snr);

    % Print Console Summary Tables
    safe_printf('\n--- %s DOMAIN SUMMARY TABLES ---\n', upper(domain));
    print_summary_table(sprintf('MSE (%s)', upper(domain)), SNRdB, m_mse, li_mse_avg, mmse_mse_avg, folder_labels);
    print_summary_table(sprintf('NMSE [dB] (%s)', upper(domain)), SNRdB, m_nmse_db, li_nmse_db_avg, mmse_nmse_db_avg, folder_labels);
    print_summary_table(sprintf('SSIM (%s)', upper(domain)), SNRdB, m_ssim, li_ssim_avg, mmse_ssim_avg, folder_labels);
    print_summary_table(sprintf('BER (%s)', upper(domain)), SNRdB, m_ber, li_ber_avg, mmse_ber_avg, folder_labels);

    % Define Color Palette & Styles for Inferred Models
    model_colors = {
        [0.8500 0.3250 0.0980], ... % Red-Orange
        [0.4660 0.6740 0.1880], ... % Green
        [0.4940 0.1840 0.5560], ... % Purple
        [0.9290 0.6940 0.1250], ... % Yellow-Orange
        [0.3010 0.7450 0.9330], ... % Cyan
        [0.6350 0.0780 0.1840], ... % Dark Red
        [0.0000 0.5000 0.5000], ... % Teal
        [0.7500 0.0000 0.7500]      % Magenta
    };
    model_markers = {'^', 'v', 'd', '*', 'p', 'h', 'x', '+'};

    % Benchmark Styles
    li_color   = [0 0.4470 0.7410]; % Blue
    li_style   = '--';
    li_marker  = 'o';

    mmse_color  = [0.15 0.15 0.15]; % Dark Grey / Black
    mmse_style  = '-.';
    mmse_marker = 's';

    % =========================================================================
    % FIGURE 1: MSE Comparison
    % =========================================================================
    fig1 = figure('Name', sprintf('MSE Comparison (%s)', domain), 'Color', 'w', 'Position', [100 100 750 520], 'Visible', 'off');
    hold on; h_lines = []; h_labels = {};

    for i = 1:num_folders
        if ~all(isnan(m_mse{i}))
            col = model_colors{mod(i-1, length(model_colors))+1};
            mrk = model_markers{mod(i-1, length(model_markers))+1};
            h = semilogy(SNRdB, m_mse{i}, '-', 'Color', col, 'LineWidth', 2.0, ...
                'Marker', mrk, 'MarkerSize', 8, 'MarkerFaceColor', col);
            h_lines = [h_lines; h]; %#ok<AGROW>
            h_labels{end+1} = folder_labels{i}; %#ok<AGROW>
        end
    end

    if ~all(isnan(li_mse_avg))
        h = semilogy(SNRdB, li_mse_avg, li_style, 'Color', li_color, 'LineWidth', 1.8, ...
            'Marker', li_marker, 'MarkerSize', 7, 'MarkerFaceColor', li_color);
        h_lines = [h_lines; h]; %#ok<AGROW>
        h_labels{end+1} = 'LI Benchmark'; %#ok<AGROW>
    end

    if ~all(isnan(mmse_mse_avg))
        h = semilogy(SNRdB, mmse_mse_avg, mmse_style, 'Color', mmse_color, 'LineWidth', 1.8, ...
            'Marker', mmse_marker, 'MarkerSize', 7, 'MarkerFaceColor', mmse_color);
        h_lines = [h_lines; h]; %#ok<AGROW>
        h_labels{end+1} = 'MMSE Benchmark'; %#ok<AGROW>
    end

    xlabel('SNR (dB)', 'FontSize', 12, 'FontWeight', 'bold');
    ylabel('Mean Squared Error (MSE)', 'FontSize', 12, 'FontWeight', 'bold');
    title(sprintf('MSE Performance Comparison (%s Domain)', upper(domain)), 'FontSize', 14, 'FontWeight', 'bold');
    grid on; box on; set(gca, 'YMinorGrid', 'on', 'FontSize', 11);
    if ~isempty(h_lines)
        legend(h_lines, h_labels, 'Location', 'northeast', 'FontSize', 10);
    end
    hold off;

    mse_pdf_path = fullfile(output_folder, sprintf('MSE_comparison_%s.pdf', domain));
    try exportgraphics(fig1, mse_pdf_path, 'ContentType', 'vector'); catch, saveas(fig1, mse_pdf_path); end
    if strcmp(domain, 'target')
        try exportgraphics(fig1, fullfile(output_folder, 'MSE_comparison.pdf'), 'ContentType', 'vector'); catch; end
    end
    safe_printf('Saved MSE plot (%s): %s\n', domain, mse_pdf_path);
    close(fig1);

    % =========================================================================
    % FIGURE 2: NMSE Comparison (dB)
    % =========================================================================
    fig2 = figure('Name', sprintf('NMSE Comparison (%s)', domain), 'Color', 'w', 'Position', [150 150 750 520], 'Visible', 'off');
    hold on; h_lines = []; h_labels = {};

    for i = 1:num_folders
        if ~all(isnan(m_nmse_db{i}))
            col = model_colors{mod(i-1, length(model_colors))+1};
            mrk = model_markers{mod(i-1, length(model_markers))+1};
            h = plot(SNRdB, m_nmse_db{i}, '-', 'Color', col, 'LineWidth', 2.0, ...
                'Marker', mrk, 'MarkerSize', 8, 'MarkerFaceColor', col);
            h_lines = [h_lines; h]; %#ok<AGROW>
            h_labels{end+1} = folder_labels{i}; %#ok<AGROW>
        end
    end

    if ~all(isnan(li_nmse_db_avg))
        h = plot(SNRdB, li_nmse_db_avg, li_style, 'Color', li_color, 'LineWidth', 1.8, ...
            'Marker', li_marker, 'MarkerSize', 7, 'MarkerFaceColor', li_color);
        h_lines = [h_lines; h]; %#ok<AGROW>
        h_labels{end+1} = 'LI Benchmark'; %#ok<AGROW>
    end

    if ~all(isnan(mmse_nmse_db_avg))
        h = plot(SNRdB, mmse_nmse_db_avg, mmse_style, 'Color', mmse_color, 'LineWidth', 1.8, ...
            'Marker', mmse_marker, 'MarkerSize', 7, 'MarkerFaceColor', mmse_color);
        h_lines = [h_lines; h]; %#ok<AGROW>
        h_labels{end+1} = 'MMSE Benchmark'; %#ok<AGROW>
    end

    xlabel('SNR (dB)', 'FontSize', 12, 'FontWeight', 'bold');
    ylabel('Normalized Mean Squared Error (NMSE) [dB]', 'FontSize', 12, 'FontWeight', 'bold');
    title(sprintf('NMSE Performance Comparison (%s Domain)', upper(domain)), 'FontSize', 14, 'FontWeight', 'bold');
    grid on; box on; set(gca, 'YMinorGrid', 'on', 'FontSize', 11);
    if ~isempty(h_lines)
        legend(h_lines, h_labels, 'Location', 'northeast', 'FontSize', 10);
    end
    hold off;

    nmse_pdf_path = fullfile(output_folder, sprintf('NMSE_comparison_%s.pdf', domain));
    try exportgraphics(fig2, nmse_pdf_path, 'ContentType', 'vector'); catch, saveas(fig2, nmse_pdf_path); end
    if strcmp(domain, 'target')
        try exportgraphics(fig2, fullfile(output_folder, 'NMSE_comparison.pdf'), 'ContentType', 'vector'); catch; end
    end
    safe_printf('Saved NMSE plot (%s): %s\n', domain, nmse_pdf_path);
    close(fig2);

    % =========================================================================
    % FIGURE 3: SSIM Comparison
    % =========================================================================
    fig3 = figure('Name', sprintf('SSIM Comparison (%s)', domain), 'Color', 'w', 'Position', [200 200 750 520], 'Visible', 'off');
    hold on; h_lines = []; h_labels = {};

    for i = 1:num_folders
        if ~all(isnan(m_ssim{i}))
            col = model_colors{mod(i-1, length(model_colors))+1};
            mrk = model_markers{mod(i-1, length(model_markers))+1};
            h = plot(SNRdB, m_ssim{i}, '-', 'Color', col, 'LineWidth', 2.0, ...
                'Marker', mrk, 'MarkerSize', 8, 'MarkerFaceColor', col);
            h_lines = [h_lines; h]; %#ok<AGROW>
            h_labels{end+1} = folder_labels{i}; %#ok<AGROW>
        end
    end

    if ~all(isnan(li_ssim_avg))
        h = plot(SNRdB, li_ssim_avg, li_style, 'Color', li_color, 'LineWidth', 1.8, ...
            'Marker', li_marker, 'MarkerSize', 7, 'MarkerFaceColor', li_color);
        h_lines = [h_lines; h]; %#ok<AGROW>
        h_labels{end+1} = 'LI Benchmark'; %#ok<AGROW>
    end

    if ~all(isnan(mmse_ssim_avg))
        h = plot(SNRdB, mmse_ssim_avg, mmse_style, 'Color', mmse_color, 'LineWidth', 1.8, ...
            'Marker', mmse_marker, 'MarkerSize', 7, 'MarkerFaceColor', mmse_color);
        h_lines = [h_lines; h]; %#ok<AGROW>
        h_labels{end+1} = 'MMSE Benchmark'; %#ok<AGROW>
    end

    xlabel('SNR (dB)', 'FontSize', 12, 'FontWeight', 'bold');
    ylabel('Structural Similarity Index (SSIM)', 'FontSize', 12, 'FontWeight', 'bold');
    title(sprintf('SSIM Performance Comparison (%s Domain)', upper(domain)), 'FontSize', 14, 'FontWeight', 'bold');
    grid on; box on; ylim([0 1.05]); set(gca, 'YMinorGrid', 'on', 'FontSize', 11);
    if ~isempty(h_lines)
        legend(h_lines, h_labels, 'Location', 'southeast', 'FontSize', 10);
    end
    hold off;

    ssim_pdf_path = fullfile(output_folder, sprintf('SSIM_comparison_%s.pdf', domain));
    try exportgraphics(fig3, ssim_pdf_path, 'ContentType', 'vector'); catch, saveas(fig3, ssim_pdf_path); end
    if strcmp(domain, 'target')
        try exportgraphics(fig3, fullfile(output_folder, 'SSIM_comparison.pdf'), 'ContentType', 'vector'); catch; end
    end
    safe_printf('Saved SSIM plot (%s): %s\n', domain, ssim_pdf_path);
    close(fig3);

    % =========================================================================
    % FIGURE 4: BER Comparison
    % =========================================================================
    fig4 = figure('Name', sprintf('BER Comparison (%s)', domain), 'Color', 'w', 'Position', [250 250 750 520], 'Visible', 'off');
    hold on; h_lines = []; h_labels = {};

    for i = 1:num_folders
        if ~all(isnan(m_ber{i}))
            col = model_colors{mod(i-1, length(model_colors))+1};
            mrk = model_markers{mod(i-1, length(model_markers))+1};
            h = semilogy(SNRdB, m_ber{i}, '-', 'Color', col, 'LineWidth', 2.0, ...
                'Marker', mrk, 'MarkerSize', 8, 'MarkerFaceColor', col);
            h_lines = [h_lines; h]; %#ok<AGROW>
            h_labels{end+1} = folder_labels{i}; %#ok<AGROW>
        end
    end

    if ~all(isnan(li_ber_avg))
        h = semilogy(SNRdB, li_ber_avg, li_style, 'Color', li_color, 'LineWidth', 1.8, ...
            'Marker', li_marker, 'MarkerSize', 7, 'MarkerFaceColor', li_color);
        h_lines = [h_lines; h]; %#ok<AGROW>
        h_labels{end+1} = 'LI Benchmark'; %#ok<AGROW>
    end

    if ~all(isnan(mmse_ber_avg))
        h = semilogy(SNRdB, mmse_ber_avg, mmse_style, 'Color', mmse_color, 'LineWidth', 1.8, ...
            'Marker', mmse_marker, 'MarkerSize', 7, 'MarkerFaceColor', mmse_color);
        h_lines = [h_lines; h]; %#ok<AGROW>
        h_labels{end+1} = 'MMSE Benchmark'; %#ok<AGROW>
    end

    xlabel('SNR (dB)', 'FontSize', 12, 'FontWeight', 'bold');
    ylabel('Bit Error Rate (BER)', 'FontSize', 12, 'FontWeight', 'bold');
    title(sprintf('BER Performance Comparison (%s Domain)', upper(domain)), 'FontSize', 14, 'FontWeight', 'bold');
    grid on; box on; set(gca, 'YMinorGrid', 'on', 'FontSize', 11);
    if ~isempty(h_lines)
        legend(h_lines, h_labels, 'Location', 'southwest', 'FontSize', 10);
    end
    hold off;

    ber_pdf_path = fullfile(output_folder, sprintf('BER_comparison_%s.pdf', domain));
    try exportgraphics(fig4, ber_pdf_path, 'ContentType', 'vector'); catch, saveas(fig4, ber_pdf_path); end
    if strcmp(domain, 'target')
        try exportgraphics(fig4, fullfile(output_folder, 'BER_comparison.pdf'), 'ContentType', 'vector'); catch; end
    end
    safe_printf('Saved BER plot (%s): %s\n', domain, ber_pdf_path);
    close(fig4);

    % =========================================================================
    % SAVE STRUCTURE & DOMAIN COMPARISON MAT FILE
    % =========================================================================
    domain_struct = struct();
    domain_struct.domain        = domain;
    domain_struct.SNRdB         = SNRdB;
    domain_struct.folders       = folders;
    domain_struct.folder_labels = folder_labels;

    domain_struct.models = struct();
    for i = 1:num_folders
        clean_lbl = matlab.lang.makeValidName(folder_labels{i});
        domain_struct.models.(clean_lbl).folder     = folders{i};
        domain_struct.models.(clean_lbl).mse        = m_mse{i};
        domain_struct.models.(clean_lbl).nmse       = m_nmse{i};
        domain_struct.models.(clean_lbl).nmse_dB    = m_nmse_db{i};
        domain_struct.models.(clean_lbl).ssim       = m_ssim{i};
        domain_struct.models.(clean_lbl).ber        = m_ber{i};
    end

    domain_struct.benchmarks = struct();
    domain_struct.benchmarks.li_mse_avg         = li_mse_avg;
    domain_struct.benchmarks.li_nmse_avg        = li_nmse_avg;
    domain_struct.benchmarks.li_nmse_db_avg     = li_nmse_db_avg;
    domain_struct.benchmarks.li_ssim_avg        = li_ssim_avg;
    domain_struct.benchmarks.li_ber_avg         = li_ber_avg;

    domain_struct.benchmarks.mmse_mse_avg       = mmse_mse_avg;
    domain_struct.benchmarks.mmse_nmse_avg      = mmse_nmse_avg;
    domain_struct.benchmarks.mmse_nmse_db_avg   = mmse_nmse_db_avg;
    domain_struct.benchmarks.mmse_ssim_avg      = mmse_ssim_avg;
    domain_struct.benchmarks.mmse_ber_avg       = mmse_ber_avg;

    % Save domain synthesized comparison MAT file
    dom_mat_path = fullfile(output_folder, sprintf('synthesized_comparison_results_%s.mat', domain));
    save(dom_mat_path, '-struct', 'domain_struct');
    safe_printf('Saved synthesized comparison MAT file (%s) to: %s\n', domain, dom_mat_path);

    % Generate Markdown Synthesis Report for this domain
    md_out_path = fullfile(output_folder, sprintf('synthesis_comparison_report_%s.md', domain));
    generate_domain_markdown_report(md_out_path, domain, SNRdB, num_folders, folder_labels, m_mse, m_nmse_db, m_ssim, m_ber, ...
        li_mse_avg, li_nmse_db_avg, li_ssim_avg, li_ber_avg, mmse_mse_avg, mmse_nmse_db_avg, mmse_ssim_avg, mmse_ber_avg);
end

%% Helper to generate domain-specific markdown report
function generate_domain_markdown_report(md_out_path, domain, SNRdB, num_folders, folder_labels, ...
    m_mse, m_nmse_db, m_ssim, m_ber, li_mse_avg, li_nmse_db_avg, li_ssim_avg, li_ber_avg, ...
    mmse_mse_avg, mmse_nmse_db_avg, mmse_ssim_avg, mmse_ber_avg)

    fid_md = fopen(md_out_path, 'w');
    if fid_md == -1
        return;
    end

    fprintf(fid_md, '# %s Domain Multi-Model Performance Synthesis Report\n\n', upper(domain));

    % 1. BER Table
    fprintf(fid_md, '## 1. Bit Error Rate (BER) Comparison\n\n');
    fprintf(fid_md, '| SNR (dB) |');
    for i = 1:num_folders, fprintf(fid_md, ' %s |', folder_labels{i}); end
    fprintf(fid_md, ' LI Benchmark | MMSE Benchmark |\n');
    fprintf(fid_md, '|:---:|%s:---:|:---:|\n', repmat(':---:|', 1, num_folders));
    for s = 1:length(SNRdB)
        fprintf(fid_md, '| %.1f |', SNRdB(s));
        for i = 1:num_folders
            if isnan(m_ber{i}(s)), fprintf(fid_md, ' N/A |'); else, fprintf(fid_md, ' %.6f |', m_ber{i}(s)); end
        end
        fprintf(fid_md, ' %.6f | %.6f |\n', li_ber_avg(s), mmse_ber_avg(s));
    end
    fprintf(fid_md, '\n');

    % 2. NMSE (dB) Table
    fprintf(fid_md, '## 2. Normalized Mean Squared Error (NMSE) [dB] Comparison\n\n');
    fprintf(fid_md, '| SNR (dB) |');
    for i = 1:num_folders, fprintf(fid_md, ' %s |', folder_labels{i}); end
    fprintf(fid_md, ' LI Benchmark | MMSE Benchmark |\n');
    fprintf(fid_md, '|:---:|%s:---:|:---:|\n', repmat(':---:|', 1, num_folders));
    for s = 1:length(SNRdB)
        fprintf(fid_md, '| %.1f |', SNRdB(s));
        for i = 1:num_folders
            if isnan(m_nmse_db{i}(s)), fprintf(fid_md, ' N/A |'); else, fprintf(fid_md, ' %.2f dB |', m_nmse_db{i}(s)); end
        end
        fprintf(fid_md, ' %.2f dB | %.2f dB |\n', li_nmse_db_avg(s), mmse_nmse_db_avg(s));
    end
    fprintf(fid_md, '\n');

    % 3. SSIM Table
    fprintf(fid_md, '## 3. Structural Similarity Index (SSIM) Comparison\n\n');
    fprintf(fid_md, '| SNR (dB) |');
    for i = 1:num_folders, fprintf(fid_md, ' %s |', folder_labels{i}); end
    fprintf(fid_md, ' LI Benchmark | MMSE Benchmark |\n');
    fprintf(fid_md, '|:---:|%s:---:|:---:|\n', repmat(':---:|', 1, num_folders));
    for s = 1:length(SNRdB)
        fprintf(fid_md, '| %.1f |', SNRdB(s));
        for i = 1:num_folders
            if isnan(m_ssim{i}(s)), fprintf(fid_md, ' N/A |'); else, fprintf(fid_md, ' %.4f |', m_ssim{i}(s)); end
        end
        fprintf(fid_md, ' %.4f | %.4f |\n', li_ssim_avg(s), mmse_ssim_avg(s));
    end
    fprintf(fid_md, '\n');

    % 4. MSE Table
    fprintf(fid_md, '## 4. Mean Squared Error (MSE) Comparison\n\n');
    fprintf(fid_md, '| SNR (dB) |');
    for i = 1:num_folders, fprintf(fid_md, ' %s |', folder_labels{i}); end
    fprintf(fid_md, ' LI Benchmark | MMSE Benchmark |\n');
    fprintf(fid_md, '|:---:|%s:---:|:---:|\n', repmat(':---:|', 1, num_folders));
    for s = 1:length(SNRdB)
        fprintf(fid_md, '| %.1f |', SNRdB(s));
        for i = 1:num_folders
            if isnan(m_mse{i}(s)), fprintf(fid_md, ' N/A |'); else, fprintf(fid_md, ' %.3e |', m_mse{i}(s)); end
        end
        fprintf(fid_md, ' %.3e | %.3e |\n', li_mse_avg(s), mmse_mse_avg(s));
    end
    fclose(fid_md);
    safe_printf('Saved comparative synthesis markdown report (%s) to: %s\n', domain, md_out_path);
end

%% Helper to generate unified multi-domain markdown report
function generate_unified_markdown_comparison_report(output_folder, comp_res, folder_labels)
    md_out_path = fullfile(output_folder, 'synthesis_comparison_report.md');
    fid_md = fopen(md_out_path, 'w');
    if fid_md == -1
        return;
    end

    fprintf(fid_md, '# CORAL UDA Multi-Model Performance Synthesis Report\n\n');
    fprintf(fid_md, 'Consolidated performance comparison across evaluated models/layers for both Target and Source domains.\n\n');

    domains = {'target', 'source'};
    for d = 1:length(domains)
        dom = domains{d};
        if isfield(comp_res, dom) && ~isempty(comp_res.(dom))
            ds = comp_res.(dom);
            SNRdB = ds.SNRdB;
            num_models = length(folder_labels);

            fprintf(fid_md, '## %s Domain Evaluation\n\n', upper(dom));

            % BER Table
            fprintf(fid_md, '### BER Performance (%s)\n', upper(dom));
            fprintf(fid_md, '| SNR (dB) |');
            for i = 1:num_models, fprintf(fid_md, ' %s |', folder_labels{i}); end
            fprintf(fid_md, ' LI Benchmark | MMSE Benchmark |\n');
            fprintf(fid_md, '|:---:|%s:---:|:---:|\n', repmat(':---:|', 1, num_models));
            for s = 1:length(SNRdB)
                fprintf(fid_md, '| %.1f |', SNRdB(s));
                for i = 1:num_models
                    clean_lbl = matlab.lang.makeValidName(folder_labels{i});
                    fprintf(fid_md, ' %.6f |', ds.models.(clean_lbl).ber(s));
                end
                fprintf(fid_md, ' %.6f | %.6f |\n', ds.benchmarks.li_ber_avg(s), ds.benchmarks.mmse_ber_avg(s));
            end
            fprintf(fid_md, '\n');

            % NMSE Table
            fprintf(fid_md, '### NMSE [dB] Performance (%s)\n', upper(dom));
            fprintf(fid_md, '| SNR (dB) |');
            for i = 1:num_models, fprintf(fid_md, ' %s |', folder_labels{i}); end
            fprintf(fid_md, ' LI Benchmark | MMSE Benchmark |\n');
            fprintf(fid_md, '|:---:|%s:---:|:---:|\n', repmat(':---:|', 1, num_models));
            for s = 1:length(SNRdB)
                fprintf(fid_md, '| %.1f |', SNRdB(s));
                for i = 1:num_models
                    clean_lbl = matlab.lang.makeValidName(folder_labels{i});
                    fprintf(fid_md, ' %.2f dB |', ds.models.(clean_lbl).nmse_dB(s));
                end
                fprintf(fid_md, ' %.2f dB | %.2f dB |\n', ds.benchmarks.li_nmse_db_avg(s), ds.benchmarks.mmse_nmse_db_avg(s));
            end
            fprintf(fid_md, '\n');
        end
    end

    fclose(fid_md);
    safe_printf('Saved unified synthesis comparison markdown report to: %s\n', md_out_path);
end

%% ========================================================================
%% GENERAL UTILITIES
%% ========================================================================

function safe_printf(fmt, varargin)
    try
        fprintf(fmt, varargin{:});
    catch
    end
end

function val = extract_field(data, explicit_keys, prefix, exclude_tokens, expected_len, is_db_field)
    if nargin < 6, is_db_field = false; end
    val = NaN(1, expected_len);

    % 1. Try explicit keys first
    for k = 1:length(explicit_keys)
        key = explicit_keys{k};
        if isfield(data, key)
            v = double(data.(key));
            if numel(v) == expected_len
                val = v(:).';
                return;
            end
        end
    end

    % 2. Search dynamic field names
    all_fields = fieldnames(data);
    for f = 1:length(all_fields)
        fn = all_fields{f};
        fn_lower = lower(fn);

        if startsWith(fn_lower, lower(prefix))
            skip = false;
            for ex = 1:length(exclude_tokens)
                if contains(fn_lower, lower(exclude_tokens{ex}))
                    skip = true;
                    break;
                end
            end
            if skip, continue; end

            if is_db_field && ~endsWith(fn_lower, 'db')
                continue;
            elseif ~is_db_field && endsWith(fn_lower, 'db')
                continue;
            end

            v = double(data.(fn));
            if numel(v) == expected_len
                val = v(:).';
                return;
            end
        end
    end
end

function avg_vec = compute_benchmark_avg(cell_data, expected_len)
    num_folders = length(cell_data);
    mat = NaN(num_folders, expected_len);
    for i = 1:num_folders
        if ~isempty(cell_data{i}) && numel(cell_data{i}) == expected_len
            mat(i, :) = cell_data{i};
        end
    end
    avg_vec = mean(mat, 1, 'omitnan');
end

function print_summary_table(metric_name, SNRdB, model_cell, li_avg, mmse_avg, folder_labels)
    num_folders = length(model_cell);
    num_snr = length(SNRdB);

    safe_printf('------------------------------------------------------------------------\n');
    safe_printf(' Summary Table: %s\n', metric_name);
    safe_printf('------------------------------------------------------------------------\n');
    safe_printf('%-10s', 'SNR (dB)');
    for i = 1:num_folders
        safe_printf('%-32s', truncate_str(folder_labels{i}, 31));
    end
    safe_printf('%-22s%-22s\n', 'LI Benchmark ', 'MMSE Benchmark ');
    safe_printf('%s\n', repmat('-', 1, 10 + 32*num_folders + 44));

    for s = 1:num_snr
        safe_printf('%-10.1f', SNRdB(s));
        for i = 1:num_folders
            val = model_cell{i}(s);
            if isnan(val)
                safe_printf('%-32s', 'N/A');
            elseif contains(metric_name, 'BER') || contains(metric_name, 'MSE') && ~contains(metric_name, 'NMSE')
                safe_printf('%-32.6e', val);
            else
                safe_printf('%-32.4f', val);
            end
        end

        % LI Avg
        if isnan(li_avg(s))
            safe_printf('%-22s', 'N/A');
        elseif contains(metric_name, 'BER') || contains(metric_name, 'MSE') && ~contains(metric_name, 'NMSE')
            safe_printf('%-22.6e', li_avg(s));
        else
            safe_printf('%-22.4f', li_avg(s));
        end

        % MMSE Avg
        if isnan(mmse_avg(s))
            safe_printf('%-22s', 'N/A');
        elseif contains(metric_name, 'BER') || contains(metric_name, 'MSE') && ~contains(metric_name, 'NMSE')
            safe_printf('%-22.6e', mmse_avg(s));
        else
            safe_printf('%-22.4f', mmse_avg(s));
        end
        safe_printf('\n');
    end
    safe_printf('\n');
end

function s_out = truncate_str(s_in, max_len)
    if length(s_in) > max_len
        s_out = [s_in(1:max_len-3) '...'];
    else
        s_out = s_in;
    end
end
