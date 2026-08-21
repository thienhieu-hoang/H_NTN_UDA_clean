%{
========================================================================================
    NTN OFDM Synthesized Comparison & Overall Plot Generation (MATLAB Version)
========================================================================================
OVERVIEW:
  This script loads pre-synthesized 'synthesized_results.mat' files from multiple model
  runs (e.g. LI+DnCNN+Attention, LI+DnCNN, LS+Attention), consolidates their metrics,
  saves a combined MAT file ('overall_synthesized_comparison.mat'), and generates
  publication-quality comparative PDF plots for MMSE/MSE, NMSE (dB), SSIM, and BER.

  BENCHMARK AVERAGING:
    The LI benchmark (LS+LI) and LMMSE benchmark metrics (MSE, NMSE, SSIM, BER) are
    automatically averaged across all loaded model approaches/datasets to provide
    consistent, unified benchmark baseline curves on all comparative plots.

USAGE:
  syn_syn_results()                           % Runs with default configs
  syn_syn_results(save_path, data_configs)     % Custom output path & dataset configs
========================================================================================
%}

if exist('mfilename', 'builtin') && ~isempty(mfilename('fullpath'))
    script_dir = fileparts(mfilename('fullpath'));
else
    script_dir = pwd;
end

save_path = 'C:\Users\AT30890\Hoctap\1_Hprediction\working\H_predict_NTN\Hest_NTN_UDA_clean\single_dataset\DUR100nsFix_2p18G_600km_70deg_r15km_20to30mps_syn';
data_configs = { ...
            struct('path', fullfile(script_dir, 'DUR100nsFix_2p18G_600km_70deg_r15km_20to30mps_LI_DnCNN_Attention', 'LI_synthesize'), ...
                   'label', 'LI+DnCNN+Attention', ...
                   'color', [0.000, 0.125, 0.376], ... % Dark Navy (#002060)
                   'marker', 'o'), ...
            struct('path', fullfile(script_dir, 'DUR100nsFix_2p18G_600km_70deg_r15km_20to30mps_LI_DnCNN', 'LI_synthesize'), ...
                   'label', 'LI+DnCNN', ...
                   'color', [0.000, 0.439, 0.753], ... % Royal Blue (#0070c0)
                   'marker', 's'), ...
            struct('path', fullfile(script_dir, 'DUR100nsFix_2p18G_600km_70deg_r15km_20to30mps_LS_Attention_standardize', 'LS_synthesize'), ...
                   'label', 'LS+Attention', ...
                   'color', [0.753, 0.000, 0.000], ... % Dark Red (#c00000)
                   'marker', '^') ...
        };
syn_syn_results(save_path, data_configs);

function syn_syn_results(save_path, data_configs)

    if exist('mfilename', 'builtin') && ~isempty(mfilename('fullpath'))
        script_dir = fileparts(mfilename('fullpath'));
    else
        script_dir = pwd;
    end

    % =========================================================================
    % 1. DEFAULT DATA CONFIGURATIONS (PATHS, LABELS, COLORS, MARKERS)
    % =========================================================================
    if nargin < 1 || isempty(save_path)
        save_path = fullfile(script_dir, 'DUR100nsFix_2p18G_600km_70deg_r15km_20to30mps_syn');
    end

    if nargin < 2 || isempty(data_configs)
        data_configs = { ...
            struct('path', fullfile(script_dir, 'DnCNN_Attention_DUR100nsFix_2p18G_600km_70deg_r15km_20to30mps_LI', 'LI_synthesize'), ...
                   'label', 'LI+DnCNN+Attention', ...
                   'color', [0.000, 0.125, 0.376], ... % Dark Navy (#002060)
                   'marker', 'o'), ...
            struct('path', fullfile(script_dir, 'Clipped_DUR100nsFix_2p18G_600km_70deg_r15km_20to30mps', 'LI_synthesize'), ...
                   'label', 'LI+DnCNN', ...
                   'color', [0.000, 0.439, 0.753], ... % Royal Blue (#0070c0)
                   'marker', 's'), ...
            struct('path', fullfile(script_dir, 'Attention_DUR100nsFix_2p18G_600km_70deg_r15km_20to30mps', 'LS_synthesize'), ...
                   'label', 'LS+Attention', ...
                   'color', [0.753, 0.000, 0.000], ... % Dark Red (#c00000)
                   'marker', '^') ...
        };
    end

    % Ensure output directory exists
    output_dir = save_path;
    if ~exist(output_dir, 'dir')
        mkdir(output_dir);
    end

    fprintf('========================================================================\n');
    fprintf('  NTN Synthesized Results Comparison & Plot Generator (MATLAB)         \n');
    fprintf('========================================================================\n');
    fprintf('Output Directory: %s\n', output_dir);
    fprintf('Configured Datasets: %d\n\n', length(data_configs));

    % =========================================================================
    % 2. LOAD AND PARSE SYNTHESIZED RESULTS (MODELS + BENCHMARKS)
    % =========================================================================
    loaded_data = {};

    for c_idx = 1:length(data_configs)
        cfg = data_configs{c_idx};
        mat_path = fullfile(cfg.path, 'synthesized_results.mat');

        if ~exist(mat_path, 'file')
            fprintf('[Warning] File not found: %s\n', mat_path);
            continue;
        end

        fprintf('Loading results from: %s\n', mat_path);
        mat = load(mat_path);

        % Extract Model Output Metrics
        snr       = extract_field(mat, {'snr', 'SNRdB', 'SNR'});
        mmse      = extract_field(mat, {'mmse_test', 'mse_test', 'mse_output', 'mse_infer'});
        nmse_db   = extract_field(mat, {'nmse_test_db', 'nmse_output_db', 'nmse_infer_dB', 'nmse_infer_db'});
        ssim_val  = extract_field(mat, {'ssim_test', 'ssim_output', 'ssim_infer'});
        ber_val   = extract_field(mat, {'ber_test', 'ber_output', 'ber_infer'}, []);

        % Extract LI Benchmark Metrics for this dataset
        mmse_li    = extract_field(mat, {'mse_li', 'mmse_input_test', 'mse_input_test'}, []);
        nmse_db_li = extract_field(mat, {'nmse_li_dB', 'nmse_li_db', 'nmse_input_test_db'}, []);
        ssim_li    = extract_field(mat, {'ssim_li', 'ssim_input_test'}, []);
        ber_li     = extract_field(mat, {'ber_li', 'ber_input_test'}, []);

        % Extract LMMSE Benchmark Metrics for this dataset
        mmse_lmmse    = extract_field(mat, {'mse_lmmse', 'mmse_lmmse'}, []);
        nmse_db_lmmse = extract_field(mat, {'nmse_lmmse_dB', 'nmse_lmmse_db'}, []);
        ssim_lmmse    = extract_field(mat, {'ssim_lmmse'}, []);
        ber_lmmse     = extract_field(mat, {'ber_lmmse'}, []);

        % Ensure color is RGB 1x3 double
        if ischar(cfg.color) || isstring(cfg.color)
            color_rgb = hex2rgb(char(cfg.color));
        else
            color_rgb = double(cfg.color);
        end

        item = struct();
        item.folder_path = cfg.path;
        item.mat_path    = mat_path;
        item.label       = cfg.label;
        item.color       = color_rgb;
        item.marker      = cfg.marker;
        item.snr         = double(snr(:).');
        item.mmse        = double(mmse(:).');
        item.nmse_db     = double(nmse_db(:).');
        item.ssim        = double(ssim_val(:).');
        item.ber         = double(ber_val(:).');

        % Benchmark fields per dataset
        item.mmse_li     = double(mmse_li(:).');
        item.nmse_db_li  = double(nmse_db_li(:).');
        item.ssim_li     = double(ssim_li(:).');
        item.ber_li      = double(ber_li(:).');

        item.mmse_lmmse    = double(mmse_lmmse(:).');
        item.nmse_db_lmmse = double(nmse_db_lmmse(:).');
        item.ssim_lmmse    = double(ssim_lmmse(:).');
        item.ber_lmmse     = double(ber_lmmse(:).');

        loaded_data{end+1} = item; %#ok<AGROW>
    end

    if isempty(loaded_data)
        error('[Error] No valid synthesized_results.mat files found to plot.');
    end

    % =========================================================================
    % 3. COMPUTE AVERAGED BENCHMARKS ACROSS ALL LOADED APPROACHES
    % =========================================================================
    li_mmse_mat = []; li_nmse_db_mat = []; li_ssim_mat = []; li_ber_mat = [];
    lmmse_mmse_mat = []; lmmse_nmse_db_mat = []; lmmse_ssim_mat = []; lmmse_ber_mat = [];

    for i = 1:length(loaded_data)
        d = loaded_data{i};

        % LI Benchmark collection
        if ~isempty(d.mmse_li)
            li_mmse_mat = [li_mmse_mat; d.mmse_li]; %#ok<AGROW>
        end
        if ~isempty(d.nmse_db_li)
            li_nmse_db_mat = [li_nmse_db_mat; d.nmse_db_li]; %#ok<AGROW>
        end
        if ~isempty(d.ssim_li)
            li_ssim_mat = [li_ssim_mat; d.ssim_li]; %#ok<AGROW>
        end
        if ~isempty(d.ber_li)
            li_ber_mat = [li_ber_mat; d.ber_li]; %#ok<AGROW>
        end

        % LMMSE Benchmark collection
        if ~isempty(d.mmse_lmmse)
            lmmse_mmse_mat = [lmmse_mmse_mat; d.mmse_lmmse]; %#ok<AGROW>
        end
        if ~isempty(d.nmse_db_lmmse)
            lmmse_nmse_db_mat = [lmmse_nmse_db_mat; d.nmse_db_lmmse]; %#ok<AGROW>
        end
        if ~isempty(d.ssim_lmmse)
            lmmse_ssim_mat = [lmmse_ssim_mat; d.ssim_lmmse]; %#ok<AGROW>
        end
        if ~isempty(d.ber_lmmse)
            lmmse_ber_mat = [lmmse_ber_mat; d.ber_lmmse]; %#ok<AGROW>
        end
    end

    % Calculate averages across approaches
    avg_li_mmse    = calculate_average_row(li_mmse_mat);
    avg_li_nmse_db = calculate_average_row(li_nmse_db_mat);
    avg_li_ssim    = calculate_average_row(li_ssim_mat);
    avg_li_ber     = calculate_average_row(li_ber_mat);

    avg_lmmse_mmse    = calculate_average_row(lmmse_mmse_mat);
    avg_lmmse_nmse_db = calculate_average_row(lmmse_nmse_db_mat);
    avg_lmmse_ssim    = calculate_average_row(lmmse_ssim_mat);
    avg_lmmse_ber     = calculate_average_row(lmmse_ber_mat);

    % =========================================================================
    % 4. SAVE COMBINED COMPARATIVE MAT FILE WITH AVERAGED BENCHMARKS
    % =========================================================================
    mat_out_path = fullfile(output_dir, 'overall_synthesized_comparison.mat');
    export_struct = struct();

    for i = 1:length(loaded_data)
        d = loaded_data{i};
        clean_label = regexprep(d.label, '\W', '_');

        export_struct.(sprintf('%s_snr', clean_label))     = d.snr;
        export_struct.(sprintf('%s_mmse', clean_label))    = d.mmse;
        export_struct.(sprintf('%s_nmse_db', clean_label)) = d.nmse_db;
        export_struct.(sprintf('%s_ssim', clean_label))    = d.ssim;
        if ~isempty(d.ber)
            export_struct.(sprintf('%s_ber', clean_label)) = d.ber;
        end
    end

    % Store averaged benchmark arrays in exported MAT
    export_struct.avg_li_mmse       = avg_li_mmse;
    export_struct.avg_li_nmse_db    = avg_li_nmse_db;
    export_struct.avg_li_ssim       = avg_li_ssim;
    export_struct.avg_li_ber        = avg_li_ber;

    export_struct.avg_lmmse_mmse    = avg_lmmse_mmse;
    export_struct.avg_lmmse_nmse_db = avg_lmmse_nmse_db;
    export_struct.avg_lmmse_ssim    = avg_lmmse_ssim;
    export_struct.avg_lmmse_ber     = avg_lmmse_ber;

    export_struct.plotted_labels = cellfun(@(x) x.label, data_configs, 'UniformOutput', false);
    export_struct.plotted_paths  = cellfun(@(x) x.path, data_configs, 'UniformOutput', false);

    save(mat_out_path, '-struct', 'export_struct');
    fprintf('\nSaved combined comparison MAT file to: %s\n', mat_out_path);

    % Save text configuration list
    txt_out_path = fullfile(output_dir, 'plotted_folders.txt');
    fid = fopen(txt_out_path, 'w');
    if fid ~= -1
        fprintf(fid, 'Folders plotted in this comparison:\n');
        for idx = 1:length(data_configs)
            fprintf(fid, '  - Label: %s\n', data_configs{idx}.label);
            fprintf(fid, '    Path:  %s\n', data_configs{idx}.path);
        end
        fclose(fid);
    end

    % Styling parameters
    LINE_WIDTH  = 2.2;
    MARKER_SIZE = 8;
    snr_axis    = loaded_data{1}.snr;

    % =========================================================================
    % PLOT 1: MMSE COMPARISON
    % =========================================================================
    fig1 = figure('Name', 'Overall MMSE Comparison', 'Color', 'w', 'Position', [100 100 750 550], 'Visible', 'off');
    hold on;
    h_lines = []; h_labels = {};

    % 1. Model Output Curves
    for i = 1:length(loaded_data)
        d = loaded_data{i};
        h = semilogy(d.snr, d.mmse, '-', 'Color', d.color, 'LineWidth', LINE_WIDTH, ...
            'Marker', d.marker, 'MarkerSize', MARKER_SIZE, 'MarkerFaceColor', d.color);
        h_lines(end+1) = h; %#ok<AGROW>
        h_labels{end+1} = d.label; %#ok<AGROW>
    end

    % 2. Averaged LI Benchmark
    if ~isempty(avg_li_mmse)
        h_li = semilogy(snr_axis, avg_li_mmse, '--o', 'Color', [0.450 0.450 0.450], ...
            'LineWidth', 1.6, 'MarkerSize', 6, 'MarkerFaceColor', [0.450 0.450 0.450]);
        h_lines(end+1) = h_li; %#ok<AGROW>
        h_labels{end+1} = 'Avg LS+LI Benchmark'; %#ok<AGROW>
    end

    % 3. Averaged LMMSE Benchmark
    if ~isempty(avg_lmmse_mmse)
        h_lmmse = semilogy(snr_axis, avg_lmmse_mmse, '--s', 'Color', [0.000 0.000 0.000], ...
            'LineWidth', 1.6, 'MarkerSize', 6, 'MarkerFaceColor', [0.000 0.000 0.000]);
        h_lines(end+1) = h_lmmse; %#ok<AGROW>
        h_labels{end+1} = 'Avg LMMSE Benchmark'; %#ok<AGROW>
    end

    xlabel('SNR (dB)', 'FontSize', 12, 'FontWeight', 'bold');
    ylabel('MMSE (log-scale)', 'FontSize', 12, 'FontWeight', 'bold');
    title('Test Set MMSE Comparison', 'FontSize', 14, 'FontWeight', 'bold');
    grid on; box on; set(gca, 'YMinorGrid', 'on', 'FontSize', 11);
    legend(h_lines, h_labels, 'Location', 'best', 'FontSize', 10);
    hold off;

    mmse_pdf_path = fullfile(output_dir, 'overall_mmse_comparison.pdf');
    save_pdf_figure(fig1, mmse_pdf_path);
    fprintf('Saved MMSE plot to: %s\n', mmse_pdf_path);
    close(fig1);

    % =========================================================================
    % PLOT 2: NMSE (dB) COMPARISON
    % =========================================================================
    fig2 = figure('Name', 'Overall NMSE Comparison', 'Color', 'w', 'Position', [150 150 750 550], 'Visible', 'off');
    hold on;
    h_lines = []; h_labels = {};

    % 1. Model Output Curves
    for i = 1:length(loaded_data)
        d = loaded_data{i};
        h = plot(d.snr, d.nmse_db, '-', 'Color', d.color, 'LineWidth', LINE_WIDTH, ...
            'Marker', d.marker, 'MarkerSize', MARKER_SIZE, 'MarkerFaceColor', d.color);
        h_lines(end+1) = h; %#ok<AGROW>
        h_labels{end+1} = d.label; %#ok<AGROW>
    end

    % 2. Averaged LI Benchmark
    if ~isempty(avg_li_nmse_db)
        h_li = plot(snr_axis, avg_li_nmse_db, '--o', 'Color', [0.450 0.450 0.450], ...
            'LineWidth', 1.6, 'MarkerSize', 6, 'MarkerFaceColor', [0.450 0.450 0.450]);
        h_lines(end+1) = h_li; %#ok<AGROW>
        h_labels{end+1} = 'Avg LS+LI Benchmark'; %#ok<AGROW>
    end

    % 3. Averaged LMMSE Benchmark
    if ~isempty(avg_lmmse_nmse_db)
        h_lmmse = plot(snr_axis, avg_lmmse_nmse_db, '--s', 'Color', [0.000 0.000 0.000], ...
            'LineWidth', 1.6, 'MarkerSize', 6, 'MarkerFaceColor', [0.000 0.000 0.000]);
        h_lines(end+1) = h_lmmse; %#ok<AGROW>
        h_labels{end+1} = 'Avg LMMSE Benchmark'; %#ok<AGROW>
    end

    xlabel('SNR (dB)', 'FontSize', 12, 'FontWeight', 'bold');
    ylabel('NMSE (dB)', 'FontSize', 12, 'FontWeight', 'bold');
    title('Test Set NMSE (dB) Comparison', 'FontSize', 14, 'FontWeight', 'bold');
    grid on; box on; set(gca, 'YMinorGrid', 'on', 'FontSize', 11);
    legend(h_lines, h_labels, 'Location', 'best', 'FontSize', 10);
    hold off;

    nmse_pdf_path = fullfile(output_dir, 'overall_nmse_comparison.pdf');
    save_pdf_figure(fig2, nmse_pdf_path);
    fprintf('Saved NMSE (dB) plot to: %s\n', nmse_pdf_path);
    close(fig2);

    % =========================================================================
    % PLOT 3: SSIM COMPARISON
    % =========================================================================
    fig3 = figure('Name', 'Overall SSIM Comparison', 'Color', 'w', 'Position', [200 200 750 550], 'Visible', 'off');
    hold on;
    h_lines = []; h_labels = {};

    % 1. Model Output Curves
    for i = 1:length(loaded_data)
        d = loaded_data{i};
        h = plot(d.snr, d.ssim, '-', 'Color', d.color, 'LineWidth', LINE_WIDTH, ...
            'Marker', d.marker, 'MarkerSize', MARKER_SIZE, 'MarkerFaceColor', d.color);
        h_lines(end+1) = h; %#ok<AGROW>
        h_labels{end+1} = d.label; %#ok<AGROW>
    end

    % 2. Averaged LI Benchmark
    if ~isempty(avg_li_ssim)
        h_li = plot(snr_axis, avg_li_ssim, '--o', 'Color', [0.450 0.450 0.450], ...
            'LineWidth', 1.6, 'MarkerSize', 6, 'MarkerFaceColor', [0.450 0.450 0.450]);
        h_lines(end+1) = h_li; %#ok<AGROW>
        h_labels{end+1} = 'Avg LS+LI Benchmark'; %#ok<AGROW>
    end

    % 3. Averaged LMMSE Benchmark
    if ~isempty(avg_lmmse_ssim)
        h_lmmse = plot(snr_axis, avg_lmmse_ssim, '--s', 'Color', [0.000 0.000 0.000], ...
            'LineWidth', 1.6, 'MarkerSize', 6, 'MarkerFaceColor', [0.000 0.000 0.000]);
        h_lines(end+1) = h_lmmse; %#ok<AGROW>
        h_labels{end+1} = 'Avg LMMSE Benchmark'; %#ok<AGROW>
    end

    xlabel('SNR (dB)', 'FontSize', 12, 'FontWeight', 'bold');
    ylabel('SSIM', 'FontSize', 12, 'FontWeight', 'bold');
    title('Test Set SSIM Comparison', 'FontSize', 14, 'FontWeight', 'bold');
    grid on; box on; ylim([0 1.05]); set(gca, 'YMinorGrid', 'on', 'FontSize', 11);
    legend(h_lines, h_labels, 'Location', 'best', 'FontSize', 10);
    hold off;

    ssim_pdf_path = fullfile(output_dir, 'overall_ssim_comparison.pdf');
    save_pdf_figure(fig3, ssim_pdf_path);
    fprintf('Saved SSIM plot to: %s\n', ssim_pdf_path);
    close(fig3);

    % =========================================================================
    % PLOT 4: BER COMPARISON
    % =========================================================================
    has_ber = false;
    for i = 1:length(loaded_data)
        if ~isempty(loaded_data{i}.ber)
            has_ber = true;
            break;
        end
    end

    if has_ber
        fig4 = figure('Name', 'Overall BER Comparison', 'Color', 'w', 'Position', [250 250 750 550], 'Visible', 'off');
        hold on;
        h_lines = []; h_labels = {};

        for i = 1:length(loaded_data)
            d = loaded_data{i};
            if ~isempty(d.ber)
                h = semilogy(d.snr, d.ber, '-', 'Color', d.color, 'LineWidth', LINE_WIDTH, ...
                    'Marker', d.marker, 'MarkerSize', MARKER_SIZE, 'MarkerFaceColor', d.color);
                h_lines(end+1) = h; %#ok<AGROW>
                h_labels{end+1} = d.label; %#ok<AGROW>
            end
        end

        % Averaged LI BER Benchmark
        if ~isempty(avg_li_ber)
            h_li = semilogy(snr_axis, avg_li_ber, '--o', 'Color', [0.450 0.450 0.450], ...
                'LineWidth', 1.6, 'MarkerSize', 6, 'MarkerFaceColor', [0.450 0.450 0.450]);
            h_lines(end+1) = h_li; %#ok<AGROW>
            h_labels{end+1} = 'Avg LS+LI Benchmark'; %#ok<AGROW>
        end

        % Averaged LMMSE BER Benchmark
        if ~isempty(avg_lmmse_ber)
            h_lmmse = semilogy(snr_axis, avg_lmmse_ber, '--s', 'Color', [0.000 0.000 0.000], ...
                'LineWidth', 1.6, 'MarkerSize', 6, 'MarkerFaceColor', [0.000 0.000 0.000]);
            h_lines(end+1) = h_lmmse; %#ok<AGROW>
            h_labels{end+1} = 'Avg LMMSE Benchmark'; %#ok<AGROW>
        end

        xlabel('SNR (dB)', 'FontSize', 12, 'FontWeight', 'bold');
        ylabel('Bit Error Rate (BER)', 'FontSize', 12, 'FontWeight', 'bold');
        title('Test Set BER Comparison', 'FontSize', 14, 'FontWeight', 'bold');
        grid on; box on; set(gca, 'YMinorGrid', 'on', 'FontSize', 11);
        legend(h_lines, h_labels, 'Location', 'southwest', 'FontSize', 10);
        hold off;

        ber_pdf_path = fullfile(output_dir, 'overall_ber_comparison.pdf');
        save_pdf_figure(fig4, ber_pdf_path);
        fprintf('Saved BER plot to: %s\n', ber_pdf_path);
        close(fig4);
    end

    % =========================================================================
    % 5. GENERATE MARKDOWN SUMMARY COMMENT REPORT (.md)
    % =========================================================================
    save_markdown_summary(output_dir, loaded_data, avg_li_nmse_db, avg_lmmse_nmse_db, ...
        avg_li_ssim, avg_lmmse_ssim, avg_li_mmse, avg_lmmse_mmse, avg_li_ber, avg_lmmse_ber);

    fprintf('\nAll plots, MAT comparisons, and Markdown summary saved to: %s\n', output_dir);
end

%% Helper function to calculate average across rows
function avg_row = calculate_average_row(matrix_in)
    if isempty(matrix_in)
        avg_row = [];
    else
        avg_row = mean(matrix_in, 1);
    end
end

%% Helper function to write Markdown Summary Report (.md)
function save_markdown_summary(output_dir, loaded_data, avg_li_nmse_db, avg_lmmse_nmse_db, ...
    avg_li_ssim, avg_lmmse_ssim, avg_li_mmse, avg_lmmse_mmse, avg_li_ber, avg_lmmse_ber)

    md_path = fullfile(output_dir, 'overall_synthesis_summary.md');
    fid = fopen(md_path, 'w');
    if fid == -1
        warning('Could not create Markdown summary report file at %s', md_path);
        return;
    end

    fprintf(fid, '# Overall Synthesized Results & Dataset Directory Notes\n\n');
    fprintf(fid, '**Generated Output Directory:**\n`%s`\n\n', output_dir);
    fprintf(fid, 'This document notes the exact source folders, file paths, visual configurations (labels, colors, markers), and metric performance summary for all datasets included in the comparative plots.\n\n');

    fprintf(fid, '> **Note on Benchmark Averaging:**\n');
    fprintf(fid, '> The **LS+LI Benchmark** and **LMMSE Benchmark** curves on the plots represent the **mean metric values averaged across all loaded model datasets/approaches** to provide a unified baseline comparison.\n\n');

    fprintf(fid, '--- \n\n');
    fprintf(fid, '## 1. Selected Folder Sources & Visual Configurations\n\n');
    fprintf(fid, '| # | Model / Curve Label | Source Synthesized Directory | MAT File Path | Color (RGB) | Marker |\n');
    fprintf(fid, '|:---:|:---|:---|:---|:---:|:---:|\n');

    for i = 1:length(loaded_data)
        d = loaded_data{i};
        color_str = sprintf('[%.3f, %.3f, %.3f]', d.color(1), d.color(2), d.color(3));
        fprintf(fid, '| %d | **%s** | `%s` | `%s` | `%s` | `%s` |\n', ...
            i, d.label, d.folder_path, d.mat_path, color_str, d.marker);
    end
    fprintf(fid, '\n');

    fprintf(fid, '--- \n\n');
    fprintf(fid, '## 2. Comparative Metric Summaries Across SNRs\n\n');

    % NMSE (dB) Table
    fprintf(fid, '### A. NMSE (dB) Comparison Table\n');
    snr_ref = loaded_data{1}.snr;
    header_str = '| SNR (dB) ';
    delim_str  = '|:---:';
    for i = 1:length(loaded_data)
        header_str = [header_str, '| ', loaded_data{i}.label, ' ']; %#ok<AGROW>
        delim_str  = [delim_str, '|:---:']; %#ok<AGROW>
    end
    if ~isempty(avg_li_nmse_db)
        header_str = [header_str, '| Avg LS+LI Bench '];
        delim_str  = [delim_str, '|:---:'];
    end
    if ~isempty(avg_lmmse_nmse_db)
        header_str = [header_str, '| Avg LMMSE Bench '];
        delim_str  = [delim_str, '|:---:'];
    end
    header_str = [header_str, '|\n'];
    delim_str  = [delim_str, '|\n'];

    fprintf(fid, '%s', header_str);
    fprintf(fid, '%s', delim_str);

    for s_idx = 1:length(snr_ref)
        line_str = sprintf('| %.1f ', snr_ref(s_idx));
        for i = 1:length(loaded_data)
            val = loaded_data{i}.nmse_db(s_idx);
            line_str = [line_str, sprintf('| %.2f dB ', val)]; %#ok<AGROW>
        end
        if ~isempty(avg_li_nmse_db)
            line_str = [line_str, sprintf('| %.2f dB ', avg_li_nmse_db(s_idx))];
        end
        if ~isempty(avg_lmmse_nmse_db)
            line_str = [line_str, sprintf('| %.2f dB ', avg_lmmse_nmse_db(s_idx))];
        end
        line_str = [line_str, '|\n']; %#ok<AGROW>
        fprintf(fid, '%s', line_str);
    end
    fprintf(fid, '\n');

    % SSIM Table
    fprintf(fid, '### B. SSIM Comparison Table\n');
    fprintf(fid, '%s', header_str);
    fprintf(fid, '%s', delim_str);
    for s_idx = 1:length(snr_ref)
        line_str = sprintf('| %.1f ', snr_ref(s_idx));
        for i = 1:length(loaded_data)
            val = loaded_data{i}.ssim(s_idx);
            line_str = [line_str, sprintf('| %.4f ', val)]; %#ok<AGROW>
        end
        if ~isempty(avg_li_ssim)
            line_str = [line_str, sprintf('| %.4f ', avg_li_ssim(s_idx))];
        end
        if ~isempty(avg_lmmse_ssim)
            line_str = [line_str, sprintf('| %.4f ', avg_lmmse_ssim(s_idx))];
        end
        line_str = [line_str, '|\n']; %#ok<AGROW>
        fprintf(fid, '%s', line_str);
    end
    fprintf(fid, '\n');

    % MMSE Table
    fprintf(fid, '### C. MMSE Comparison Table\n');
    fprintf(fid, '%s', header_str);
    fprintf(fid, '%s', delim_str);
    for s_idx = 1:length(snr_ref)
        line_str = sprintf('| %.1f ', snr_ref(s_idx));
        for i = 1:length(loaded_data)
            val = loaded_data{i}.mmse(s_idx);
            line_str = [line_str, sprintf('| %.3e ', val)]; %#ok<AGROW>
        end
        if ~isempty(avg_li_mmse)
            line_str = [line_str, sprintf('| %.3e ', avg_li_mmse(s_idx))];
        end
        if ~isempty(avg_lmmse_mmse)
            line_str = [line_str, sprintf('| %.3e ', avg_lmmse_mmse(s_idx))];
        end
        line_str = [line_str, '|\n']; %#ok<AGROW>
        fprintf(fid, '%s', line_str);
    end
    fprintf(fid, '\n');

    % BER Table (if available)
    has_ber = false;
    for i = 1:length(loaded_data)
        if ~isempty(loaded_data{i}.ber)
            has_ber = true;
            break;
        end
    end

    if has_ber
        fprintf(fid, '### D. BER Comparison Table\n');
        fprintf(fid, '%s', header_str);
        fprintf(fid, '%s', delim_str);
        for s_idx = 1:length(snr_ref)
            line_str = sprintf('| %.1f ', snr_ref(s_idx));
            for i = 1:length(loaded_data)
                if ~isempty(loaded_data{i}.ber)
                    val = loaded_data{i}.ber(s_idx);
                    line_str = [line_str, sprintf('| %.6f ', val)]; %#ok<AGROW>
                else
                    line_str = [line_str, '| N/A ']; %#ok<AGROW>
                end
            end
            if ~isempty(avg_li_ber)
                line_str = [line_str, sprintf('| %.6f ', avg_li_ber(s_idx))];
            end
            if ~isempty(avg_lmmse_ber)
                line_str = [line_str, sprintf('| %.6f ', avg_lmmse_ber(s_idx))];
            end
            line_str = [line_str, '|\n']; %#ok<AGROW>
            fprintf(fid, '%s', line_str);
        end
        fprintf(fid, '\n');
    end

    fclose(fid);
    fprintf('Saved Markdown summary report to: %s\n', md_path);
end

%% Helper function to extract field adaptively from MAT struct
function val = extract_field(s, field_candidates, default_val)
    if nargin < 3
        default_val = [];
    end
    val = default_val;
    for k = 1:length(field_candidates)
        fn = field_candidates{k};
        if isfield(s, fn)
            val = s.(fn);
            return;
        end
    end
end

%% Helper function to convert Hex color string to RGB 1x3 vector [0 1]
function rgb = hex2rgb(hex_str)
    hex_str = strrep(hex_str, '#', '');
    if length(hex_str) == 6
        r = hex2dec(hex_str(1:2)) / 255;
        g = hex2dec(hex_str(3:4)) / 255;
        b = hex2dec(hex_str(5:6)) / 255;
        rgb = [r, g, b];
    else
        rgb = [0.2 0.2 0.2];
    end
end

%% Helper function to safely export PDF figures
function save_pdf_figure(fig_handle, pdf_path)
    try
        exportgraphics(fig_handle, pdf_path, 'ContentType', 'vector');
    catch
        try
            saveas(fig_handle, pdf_path);
        catch
            warning('Could not export PDF figure to %s', pdf_path);
        end
    end
end
