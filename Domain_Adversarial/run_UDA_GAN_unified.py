import tensorflow as tf
import os
import sys
import numpy as np
import scipy
import ot
from scipy.io import savemat, loadmat
import h5py
import time
import argparse

# =====================================================================
# DEFAULT PARAMETERS (Used when running directly in IDE without CLI arguments)
# =====================================================================
DEFAULT_SNR = 0                 # Choices: -15, -10, -5, 0, 5, 10
DEFAULT_TYPE = 'LI'             # Choices: 'LI', 'LS', 'Prac'
DEFAULT_SOURCE = 'TDL_A_300_sim'
DEFAULT_TARGET = 'TDL_B_100_300_sim'
# =====================================================================

# Setup project directories and paths
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
helper_dir = os.path.join(current_dir, 'helper')

if project_root not in sys.path:
    sys.path.insert(0, project_root)
if helper_dir not in sys.path:
    sys.path.insert(0, helper_dir)

# Import local helpers
import utils_GAN
import PAD
import utils
import loader
import plotfig

def main():
    parser = argparse.ArgumentParser(description="Unified Domain Adversarial GAN training runner.")
    parser.add_argument('--snr', type=int, default=DEFAULT_SNR, choices=[-15, -10, -5, 0, 5, 10], help="Channel SNR in dB")
    parser.add_argument('--type', type=str, default=DEFAULT_TYPE, choices=['LI', 'LS', 'Prac'], help="Type tag mapping to sub_folder")
    parser.add_argument('--source-tdl', type=str, default=DEFAULT_SOURCE, help="Source TDL dataset name (e.g. TDL_A_300_sim)")
    parser.add_argument('--target-tdl', type=str, default=DEFAULT_TARGET, help="Target TDL dataset name (e.g. TDL_B_100_300_sim)")
    parser.add_argument('--norm-approach', type=str, default='minmax', choices=['minmax', 'std', 'no'], help="Normalization approach")
    parser.add_argument('--lower-range', type=int, default=-1, choices=[0, -1], help="Scaling range for minmax")
    parser.add_argument('--adv-weight', type=float, default=0.005, help="GAN adversarial loss weight")
    parser.add_argument('--est_weight', type=float, default=1.0, help="Estimation loss weight (main task)")
    parser.add_argument('--domain-weight', type=float, default=0.5, help="Domain classification loss weight")
    parser.add_argument('--gen-lr', type=float, default=1e-4, help="Generator learning rate")
    parser.add_argument('--disc-lr', type=float, default=1e-5, help="Discriminator learning rate")
    parser.add_argument('--domain-lr', type=float, default=5e-5, help="Domain discriminator learning rate")
    parser.add_argument('--normalized', action='store_true', help="Use normalized train step")
    parser.add_argument('--n-epochs', type=int, default=300, help="Number of training epochs")
    parser.add_argument('--batch-size', type=int, default=16, help="Batch size")
    parser.add_argument('--test-code', action='store_true', help="Run with subset of data (size 96) for testing code")
    parser.add_argument('--save-model', type=bool, default=True, help="Save trained model weights")
    parser.add_argument('--load-checkpoint', action='store_true', help="Continue training from a saved checkpoint")
    parser.add_argument('--start-epoch', type=int, default=0, help="Start epoch if continuing from checkpoint")
    parser.add_argument('--save-dir', type=str, default='', help="Custom folder directory to save results")

    args = parser.parse_args()

    SNR = args.snr
    sub_folder_map = {'LI': 'GAN_linear', 'LS': 'GAN_ls', 'Prac': 'GAN_practical'}
    sub_folder = sub_folder_map[args.type]

    # Resolve dataset paths relative to project root
    source_data_file_path = os.path.abspath(os.path.join(project_root, 'generatedChan', 'MATLAB', args.source_tdl, f'SNR_{SNR}dB', 'matlabNTN.mat'))
    target_data_file_path = os.path.abspath(os.path.join(project_root, 'generatedChan', 'MATLAB', args.target_tdl, f'SNR_{SNR}dB', 'matlabNTN.mat'))

    print(f"Loading Source dataset: {source_data_file_path}")
    print(f"Loading Target dataset: {target_data_file_path}")

    # Build save path
    if args.save_dir:
        path_temp = os.path.abspath(args.save_dir)
    else:
        def clean_tdl(name):
            return name.replace('TDL_', '').replace('_sim', '').replace('_simple', '').replace('_', '')
        src_short = clean_tdl(args.source_tdl)
        tgt_short = clean_tdl(args.target_tdl)
        path_temp = os.path.join(current_dir, 'model', f'GAN_cal_{src_short}_{tgt_short}', f'{SNR}_dB')

    os.makedirs(path_temp, exist_ok=True)
    idx_save_path = loader.find_incremental_filename(path_temp, 'ver', '_', '')
    model_path = os.path.join(path_temp, f'ver{idx_save_path}_')

    print(f"Model outputs will be saved to: {model_path}")

    # ============ Load Source and Target data ==============
    source_file = h5py.File(source_data_file_path, 'r')
    H_true_source = source_file['H_perfect']
    N_samp_source = H_true_source.shape[0]

    target_file = h5py.File(target_data_file_path, 'r')
    H_true_target = target_file['H_perfect']
    N_samp_target = H_true_target.shape[0]

    # Split dataset reproducibly
    rng_state = np.random.get_state()
    np.random.seed(1234)
    indices_source = np.arange(N_samp_source)
    np.random.shuffle(indices_source)
    indices_target = np.arange(N_samp_target)
    np.random.shuffle(indices_target)
    np.random.set_state(rng_state)

    train_size = int(np.floor(N_samp_source * 0.9) // args.batch_size * args.batch_size)
    val_size = N_samp_source - train_size

    # Repeat indices to match maximum sample count
    N_samp = max(N_samp_source, N_samp_target)
    indices_source = np.resize(indices_source, N_samp)
    indices_target = np.resize(indices_target, N_samp)

    if args.test_code:
        indices_train_source = indices_source[:96]
        indices_val_source = indices_source[2032:]
        indices_train_target = indices_target[:96]
        indices_val_target = indices_target[2032:]
        train_size = 96
        val_size = indices_val_source.shape[0]
    else:
        indices_train_source = indices_source[:train_size]
        indices_val_source = indices_source[train_size:train_size + val_size]
        indices_train_target = indices_target[:train_size]
        indices_val_target = indices_target[train_size:train_size + val_size]

    class DataLoaders:
        def __init__(self, file, indices_train, indices_val, tag, batch_size):
            self.true_train = utils.H5BatchLoader(file, dataset_name='H_perfect', batch_size=batch_size, shuffled_indices=indices_train)
            self.true_val = utils.H5BatchLoader(file, dataset_name='H_perfect', batch_size=batch_size, shuffled_indices=indices_val)
            self.input_train = utils.H5BatchLoader(file, f'H_{tag}', batch_size=batch_size, shuffled_indices=indices_train)
            self.input_val = utils.H5BatchLoader(file, f'H_{tag}', batch_size=batch_size, shuffled_indices=indices_val)

    class_dict_source = {
        'GAN_practical': DataLoaders(source_file, indices_train_source, indices_val_source, 'prac', args.batch_size),
        'GAN_linear': DataLoaders(source_file, indices_train_source, indices_val_source, 'li', args.batch_size),
        'GAN_ls': DataLoaders(source_file, indices_train_source, indices_val_source, 'ls', args.batch_size)
    }

    class_dict_target = {
        'GAN_practical': DataLoaders(target_file, indices_train_target, indices_val_target, 'prac', args.batch_size),
        'GAN_linear': DataLoaders(target_file, indices_train_target, indices_val_target, 'li', args.batch_size),
        'GAN_ls': DataLoaders(target_file, indices_train_target, indices_val_target, 'ls', args.batch_size)
    }

    loss_fn_ce = tf.keras.losses.MeanSquaredError()
    loss_fn_bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    loss_fn_domain = tf.keras.losses.BinaryCrossentropy()

    load_checkpoint = args.load_checkpoint
    start_epoch = args.start_epoch

    if load_checkpoint:
        epoch_load = start_epoch - 1
        model_path_load = os.path.join(path_temp, f'ver{idx_save_path-1}_')
    else:
        epoch_load = 0
        model_path_load = model_path

    start_time = time.perf_counter()

    loader_H_true_train_source = class_dict_source[sub_folder].true_train
    loader_H_input_train_source = class_dict_source[sub_folder].input_train
    loader_H_true_val_source = class_dict_source[sub_folder].true_val
    loader_H_input_val_source = class_dict_source[sub_folder].input_val

    loader_H_true_train_target = class_dict_target[sub_folder].true_train
    loader_H_input_train_target = class_dict_target[sub_folder].input_train
    loader_H_true_val_target = class_dict_target[sub_folder].true_val
    loader_H_input_val_target = class_dict_target[sub_folder].input_val

    # Create Distribution Plots before training
    os.makedirs(os.path.join(model_path, sub_folder, "Distribution"), exist_ok=True)
    plotfig.plotHist(loader_H_input_train_source, fig_show=False, save_path=f"{model_path}/{sub_folder}/Distribution/", name='source_beforeTrain', percent=99)
    plotfig.plotHist(loader_H_input_train_target, fig_show=False, save_path=f"{model_path}/{sub_folder}/Distribution/", name='target_beforeTrain', percent=99)

    # Initial metrics computation
    if not load_checkpoint:
        print("Calculating initial metrics...")
        w_dist = [plotfig.wasserstein_approximate(loader_H_input_train_source, loader_H_input_train_target)]
        pad_svm = PAD.original_PAD(loader_H_input_train_source, loader_H_input_train_target)
        print(f"Initial SVM PAD = {pad_svm:.4f}")

        X_features_, y_features_ = PAD.extract_features_with_pca(loader_H_input_train_source, loader_H_input_train_target, pca_components=100)
        pad_pca_svm = [PAD.calc_pad_svm(X_features_, y_features_)]
        pad_pca_lda = [PAD.calc_pad_lda(X_features_, y_features_)]
        pad_pca_logreg = [PAD.calc_pad_logreg(X_features_, y_features_)]
    else:
        # Loaded metrics placeholder
        w_dist = []
        pad_pca_svm = []
        pad_pca_lda = []
        pad_pca_logreg = []
        pad_svm = 0

    # Build model (assume n_subc based on first data sample channel dimension)
    sample_H = next(loader_H_input_train_source.generator())
    n_subc = sample_H.shape[1]
    print(f"Detected channel subcarriers (n_subc): {n_subc}")

    if not load_checkpoint:
        model = utils_GAN.GAN(n_subc=n_subc, gen_l2=None, disc_l2=1e-5)
        model_domain = utils_GAN.DomainDisc()
        gen_optimizer = tf.keras.optimizers.Adam(learning_rate=args.gen_lr, beta_1=0.5, beta_2=0.9)
        disc_optimizer = tf.keras.optimizers.Adam(learning_rate=args.disc_lr, beta_1=0.5, beta_2=0.9)
        domain_optimizer = tf.keras.optimizers.Adam(learning_rate=args.domain_lr)
        optimizer = [gen_optimizer, disc_optimizer, domain_optimizer]

        epoc_pad = []
        train_loss = []
        train_est_loss = []
        train_disc_loss = []
        train_domain_loss = []
        train_est_loss_target = []
        val_loss, val_gan_disc_loss, val_domain_disc_loss, val_est_loss_source, val_est_loss_target, val_est_loss = [], [], [], [], [], []
        source_acc, target_acc, acc, nmse_val_source, nmse_val_target, nmse_val = [], [], [], [], [], []
    else:
        model = utils_GAN.GAN(n_subc=n_subc, gen_l2=None, disc_l2=1e-5)
        dummy_input = tf.random.normal((args.batch_size, n_subc, 14, 2))
        _ = model(dummy_input)
        model_domain = utils_GAN.DomainDisc()
        dummy_input_domain = tf.random.normal((args.batch_size, 7, 14, 256))
        _ = model_domain(dummy_input_domain)

        print(f"Loading checkpoint from epoch {epoch_load+1}...")
        gen_optimizer, disc_optimizer, domain_optimizer = utils_GAN.load_checkpoint(
            model, model_path_load, sub_folder, epoch_load, domain_model=model_domain, domain_weight=args.domain_weight
        )
        optimizer = [gen_optimizer, disc_optimizer, domain_optimizer]

        loadmat_params = loadmat(f"{model_path_load}/{sub_folder}/performance/performance.mat")
        train_loss = loadmat_params['train_loss'].flatten().tolist()[:start_epoch]
        train_est_loss = loadmat_params['train_est_loss'].flatten().tolist()[:start_epoch]
        train_disc_loss = loadmat_params['train_disc_loss'].flatten().tolist()[:start_epoch]
        train_domain_loss = loadmat_params['train_domain_loss'].flatten().tolist()[:start_epoch]
        train_est_loss_target = loadmat_params['train_est_loss_target'].flatten().tolist()[:start_epoch]

        val_loss = loadmat_params['val_loss'].flatten().tolist()[:start_epoch]
        val_gan_disc_loss = loadmat_params['val_gan_disc_loss'].flatten().tolist()[:start_epoch]
        val_domain_disc_loss = loadmat_params['val_domain_disc_loss'].flatten().tolist()[:start_epoch]
        val_est_loss_source = loadmat_params['val_est_loss_source'].flatten().tolist()[:start_epoch]
        val_est_loss_target = loadmat_params['val_est_loss_target'].flatten().tolist()[:start_epoch]
        val_est_loss = loadmat_params['val_est_loss'].flatten().tolist()[:start_epoch]
        source_acc = loadmat_params['source_acc'].flatten().tolist()[:start_epoch]
        target_acc = loadmat_params['target_acc'].flatten().tolist()[:start_epoch]
        acc = loadmat_params['acc'].flatten().tolist()[:start_epoch]
        nmse_val_source = loadmat_params['nmse_val_source'].flatten().tolist()[:start_epoch]
        nmse_val_target = loadmat_params['nmse_val_target'].flatten().tolist()[:start_epoch]
        nmse_val = loadmat_params['nmse_val'].flatten().tolist()[:start_epoch]

        epoc_pad = loadmat_params['epoc_pad'].flatten().tolist()
        pad_pca_lda = loadmat_params['pad_pca_lda'].flatten().tolist()
        pad_pca_logreg = loadmat_params['pad_pca_logreg'].flatten().tolist()
        pad_pca_svm = loadmat_params['pad_pca_svm'].flatten().tolist()
        pad_svm = loadmat_params['pad_svm']

    # Sub-folder directory setup
    os.makedirs(os.path.join(model_path, sub_folder), exist_ok=True)

    flag = 1
    H_to_save = {}
    linear_interp = False

    epoch_min = 20
    epoch_step = 20

    # Main training loop
    for epoch in range(start_epoch, args.n_epochs):
        loader_H_true_train_source.reset()
        loader_H_input_train_source.reset()
        loader_H_true_train_target.reset()
        loader_H_input_train_target.reset()

        loader_H = [loader_H_input_train_source, loader_H_true_train_source, loader_H_input_train_target, loader_H_true_train_target]
        loss_fn = [loss_fn_ce, loss_fn_bce, loss_fn_domain]

        if epoch in [int(args.n_epochs * r) for r in [0, 0.25, 0.5, 0.75]] or epoch == args.n_epochs - 1:
            return_features = True
            epoc_pad.append(epoch)
        else:
            return_features = False

        # Select standard or normalized train step
        if args.normalized:
            train_step_output = utils_GAN.train_step_wgan_gp_normalized(
                model, model_domain, loader_H, loss_fn, optimizer, lower_range=args.lower_range,
                adv_weight=args.adv_weight, est_weight=args.est_weight, domain_weight=args.domain_weight,
                return_features=return_features, linear_interp=linear_interp
            )
        else:
            train_step_output = utils_GAN.train_step_wgan_gp(
                model, model_domain, loader_H, loss_fn, optimizer, lower_range=args.lower_range,
                adv_weight=args.adv_weight, est_weight=args.est_weight, domain_weight=args.domain_weight,
                return_features=return_features, linear_interp=linear_interp
            )

        train_epoc_loss_est = train_step_output.avg_epoc_loss_est
        train_epoc_loss_d = train_step_output.avg_epoc_loss_d
        train_epoc_loss_domain = train_step_output.avg_epoc_loss_domain
        train_epoc_loss = train_step_output.avg_epoc_loss
        train_epoc_loss_est_target = train_step_output.avg_epoc_loss_est_target

        print(f"Epoch {epoch+1}/{args.n_epochs} | Time: {time.perf_counter() - start_time:.2f}s")
        print(f"  Loss: {train_epoc_loss:.6f} | Est Loss (Src): {train_epoc_loss_est:.6f} | Domain Loss: {train_epoc_loss_domain:.6f}")

        # Compute PAD metrics for extracted features
        if return_features and (args.domain_weight != 0):
            features_source_file = "features_source.h5"
            features_target_file = "features_target.h5"

            X_features, y_features = PAD.extract_features_with_pca(features_source_file, features_target_file, pca_components=100)
            pad_pca_svm.append(PAD.calc_pad_svm(X_features, y_features))
            pad_pca_lda.append(PAD.calc_pad_lda(X_features, y_features))
            pad_pca_logreg.append(PAD.calc_pad_logreg(X_features, y_features))

            plotfig.plotHist(features_source_file, fig_show=False, save_path=f"{model_path}/{sub_folder}/Distribution/", name=f'source_epoch_{epoch+1}', percent=99)
            plotfig.plotHist(features_target_file, fig_show=False, save_path=f"{model_path}/{sub_folder}/Distribution/", name=f'target_epoch_{epoch+1}', percent=99)

            if os.path.exists(features_source_file):
                os.remove(features_source_file)
            if os.path.exists(features_target_file):
                os.remove(features_target_file)

        train_loss.append(train_epoc_loss)
        train_est_loss.append(train_epoc_loss_est)
        train_disc_loss.append(train_epoc_loss_d)
        train_domain_loss.append(train_epoc_loss_domain)
        train_est_loss_target.append(train_epoc_loss_est_target)

        # ===================== Evaluation =====================
        loader_H_true_val_source.reset()
        loader_H_input_val_source.reset()
        loader_H_true_val_target.reset()
        loader_H_input_val_target.reset()
        loader_H_eval = [loader_H_input_val_source, loader_H_true_val_source, loader_H_input_val_target, loader_H_true_val_target]

        if (epoch == epoch_min) or (epoch + 1 > epoch_min and (epoch - epoch_min) % epoch_step == 0):
            H_sample, epoc_val_return = utils_GAN.val_step_wgan_gp(
                model, model_domain, loader_H_eval, loss_fn, args.lower_range,
                adv_weight=args.adv_weight, est_weight=args.est_weight, domain_weight=args.domain_weight, linear_interp=linear_interp
            )
            utils_GAN.visualize_H(H_sample, H_to_save, epoch, plotfig.figChan, flag, model_path, sub_folder, domain_weight=args.domain_weight)
            flag = 0
        elif epoch == args.n_epochs - 1:
            _, epoc_val_return, H_val_gen = utils_GAN.val_step_wgan_gp(
                model, model_domain, loader_H_eval, loss_fn, args.lower_range,
                adv_weight=args.adv_weight, est_weight=args.est_weight, domain_weight=args.domain_weight,
                linear_interp=linear_interp, return_H_gen=True
            )
        else:
            _, epoc_val_return = utils_GAN.val_step_wgan_gp(
                model, model_domain, loader_H_eval, loss_fn, args.lower_range,
                adv_weight=args.adv_weight, est_weight=args.est_weight, domain_weight=args.domain_weight, linear_interp=linear_interp
            )

        utils_GAN.post_val(
            epoc_val_return, epoch, args.n_epochs, val_est_loss, val_est_loss_source, val_loss, val_est_loss_target,
            val_gan_disc_loss, val_domain_disc_loss, nmse_val_source, nmse_val_target, nmse_val, source_acc, target_acc, acc, domain_weight=args.domain_weight
        )

        # Save Checkpoint
        if (epoch == epoch_min) or (epoch + 1 > epoch_min and (epoch - epoch_min) % epoch_step == 0) or epoch == args.n_epochs - 1:
            utils_GAN.save_checkpoint(
                model, args.save_model, model_path, sub_folder, epoch, plotfig.figLoss, savemat,
                train_loss, train_est_loss, train_domain_loss, train_est_loss_target,
                val_est_loss, val_est_loss_source, val_loss, val_est_loss_target, val_gan_disc_loss, val_domain_disc_loss,
                source_acc, target_acc, acc, nmse_val_source, nmse_val_target, nmse_val,
                pad_pca_svm, pad_pca_lda, pad_pca_logreg, epoc_pad, pad_svm, train_disc_loss,
                domain_weight=args.domain_weight, optimizer=optimizer, domain_model=model_domain
            )

    # Save visual outputs
    os.makedirs(os.path.join(model_path, sub_folder, 'H_visualize'), exist_ok=True)
    savemat(os.path.join(model_path, sub_folder, 'H_visualize', 'H_trix.mat'), H_to_save)
    savemat(os.path.join(model_path, sub_folder, 'H_visualize', 'H_val_generated.mat'), {
        'H_val_gen': H_val_gen,
        'indices_val_source': indices_val_source,
        'indices_val_target': indices_val_target
    })

    print(f"Unified run completed successfully! Results stored in: {model_path}/{sub_folder}")

if __name__ == '__main__':
    main()
