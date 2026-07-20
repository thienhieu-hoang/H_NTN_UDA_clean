import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np

# Optional imports for loading .mat files
try:
    import scipy.io as sio
except ImportError:
    sio = None

try:
    import h5py
except ImportError:
    h5py = None


# =============================================================================
# 1. HA02 MODEL ARCHITECTURE (Transformer Encoder + Residual Conv Decoder)
# =============================================================================

class TransformerEncoderBlock(nn.Module):
    """
    Transformer Encoder Block for HA02 as described in Section IV-A.
    - Multi-Head Self-Attention on pilot features (N_heads = N_pilot = 2).
    - Add & Layer Normalization.
    - Feed-Forward Network (FC -> GeLU -> FC).
    - Add & Layer Normalization.
    """
    def __init__(self, num_pilot_elems=72, num_channels=2, num_heads=2):
        super(TransformerEncoderBlock, self).__init__()
        self.num_pilot_elems = num_pilot_elems  # N_pilot * Nf / 2 = 2 * 36 = 72
        self.num_channels = num_channels        # 2 (Real, Imaginary)
        self.num_heads = num_heads              # N_heads = N_pilot = 2
        
        in_dim = num_pilot_elems * num_channels # 72 * 2 = 144
        
        # FC1: resizes input from (144) to 3 * 144 = 432 for Q, K, V projection
        self.fc1 = nn.Linear(in_dim, 3 * in_dim)
        
        # FC2: projects concatenated multi-head attention back to in_dim
        self.fc2 = nn.Linear(in_dim, in_dim)
        
        # Layer Normalization 1 & 2
        self.ln1 = nn.LayerNorm(in_dim)
        self.ln2 = nn.LayerNorm(in_dim)
        
        # Feed-Forward Network: FC -> GeLU -> FC
        self.ffn = nn.Sequential(
            nn.Linear(in_dim, in_dim * 2),
            nn.GELU(),
            nn.Linear(in_dim * 2, in_dim)
        )

    def forward(self, x):
        """
        Input shape: (batch_size, num_pilot_elems, num_channels) e.g., (B, 72, 2)
        """
        B = x.shape[0]
        x_flat = x.view(B, -1)  # (B, 144)
        
        # Linear projection to generate Q, K, V
        qkv = self.fc1(x_flat)  # (B, 432)
        qkv = qkv.view(B, 3, self.num_heads, -1)  # (B, 3, 2, 72)
        Q, K, V = qkv[:, 0], qkv[:, 1], qkv[:, 2]  # Each is (B, num_heads, head_dim)
        
        # Scaled Dot-Product Attention per head
        # scale factor: sqrt(N_f / 2) = sqrt(36) = 6
        scale = np.sqrt(self.num_pilot_elems / self.num_heads)
        scores = torch.matmul(Q.unsqueeze(-1), K.unsqueeze(-2)) / scale  # (B, num_heads, head_dim, head_dim)
        attn_weights = F.softmax(scores, dim=-1)
        attn_out = torch.matmul(attn_weights, V.unsqueeze(-1)).squeeze(-1) # (B, num_heads, head_dim)
        
        # Concatenate heads and pass through second FC layer
        attn_out_flat = attn_out.view(B, -1)  # (B, 144)
        attn_proj = self.fc2(attn_out_flat)   # (B, 144)
        
        # Residual + Add & Norm 1
        x_norm1 = self.ln1(x_flat + attn_proj)
        
        # Feed-Forward Network + Add & Norm 2
        ffn_out = self.ffn(x_norm1)
        out = self.ln2(x_norm1 + ffn_out)      # (B, 144)
        
        # Reshape back to (B, 72, 2)
        out = out.view(B, self.num_pilot_elems, self.num_channels)
        return out


class ResidualConvDecoderBlock(nn.Module):
    """
    Residual Convolutional Architecture for HA02 as described in Section IV-B.
    - Conv1 (2x2 kernel, N_filter channels)
    - Residual Module (Conv -> ReLU -> Conv + Add & Norm)
    - Upsampling Module (FC layer resizing 72 -> 1008 + 1D Conv layer)
    """
    def __init__(self, num_pilot_elems=72, total_grid_elems=1008, n_filter=2):
        super(ResidualConvDecoderBlock, self).__init__()
        self.num_pilot_elems = num_pilot_elems  # 72
        self.total_grid_elems = total_grid_elems  # N_s * N_f = 14 * 72 = 1008
        self.n_filter = n_filter                  # 2 filters
        
        # First Conv Layer (operates on 2D feature map [72, 2])
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=n_filter, kernel_size=(2, 2), padding='same')
        
        # Residual Block: Conv -> ReLU -> Conv
        self.res_conv1 = nn.Conv2d(in_channels=n_filter, out_channels=n_filter, kernel_size=(2, 2), padding='same')
        self.relu = nn.ReLU()
        self.res_conv2 = nn.Conv2d(in_channels=n_filter, out_channels=n_filter, kernel_size=(2, 2), padding='same')
        self.norm = nn.BatchNorm2d(n_filter)
        
        # FC Layer for Upsampling from pilot dimension (72) to full slot dimension (1008 = 14 * 72)
        self.fc_upsample = nn.Linear(num_pilot_elems, total_grid_elems)
        
        # Final Convolutional layer (maps N_filter -> 1 channel to restore 2D structure)
        self.conv_out = nn.Conv2d(in_channels=n_filter, out_channels=1, kernel_size=(2, 2), padding='same')

    def forward(self, x):
        """
        Input shape: (B, 72, 2) from Transformer Encoder
        """
        B = x.shape[0]
        
        # Format as 2D spatial feature map: (B, 1, 72, 2)
        x_img = x.unsqueeze(1)
        
        # Conv 1
        h1 = self.conv1(x_img)  # (B, n_filter, 72, 2)
        
        # Residual Block
        res = self.res_conv1(h1)
        res = self.relu(res)
        res = self.res_conv2(res)
        h2 = self.norm(h1 + res)  # (B, n_filter, 72, 2)
        
        # Upsampling via FC layer along spatial pilot axis (dim=2)
        # Reshape to (B, n_filter * 2, 72) for FC projection across the 72 elements
        h2_perm = h2.permute(0, 1, 3, 2)  # (B, n_filter, 2, 72)
        h2_upsampled = self.fc_upsample(h2_perm) # (B, n_filter, 2, 1008)
        h2_upsampled = h2_upsampled.permute(0, 1, 3, 2) # (B, n_filter, 1008, 2)
        
        # Final Conv layer
        out = self.conv_out(h2_upsampled)  # (B, 1, 1008, 2)
        out = out.squeeze(1)               # (B, 1008, 2)
        
        # Reshape to 2D channel grid (B, 14, 72, 2) corresponding to (N_s, N_f, Real/Imag)
        out_grid = out.view(B, 14, 72, 2)
        return out_grid


class HA02Model(nn.Module):
    """
    Complete HA02 Hybrid Architecture combining:
    1. Transformer Encoder Stack (Attention pre-processor for sparse LS pilots)
    2. Residual Convolutional Architecture (Decoder + Upsampler to full 14x72 grid)
    """
    def __init__(self, num_pilot_elems=72, total_grid_elems=1008, num_channels=2, num_heads=2, n_filter=2):
        super(HA02Model, self).__init__()
        self.encoder = TransformerEncoderBlock(
            num_pilot_elems=num_pilot_elems, 
            num_channels=num_channels, 
            num_heads=num_heads
        )
        self.decoder = ResidualConvDecoderBlock(
            num_pilot_elems=num_pilot_elems, 
            total_grid_elems=total_grid_elems, 
            n_filter=n_filter
        )

    def forward(self, x):
        """
        Input:  (batch_size, 72, 2)  -- Sparse LS estimates at pilot locations (Real & Imag)
        Output: (batch_size, 14, 72, 2) -- Reconstructed full channel grid (14 symbols x 72 subcarriers x 2)
        """
        encoder_out = self.encoder(x)
        full_grid_out = self.decoder(encoder_out)
        return full_grid_out


# =============================================================================
# 2. HUBER LOSS FUNCTION
# =============================================================================

class HuberLoss(nn.Module):
    """
    Huber Loss with transition threshold delta = 1.0 (Equation 8 in the paper).
    Less sensitive to outliers than MSE.
    """
    def __init__(self, delta=1.0):
        super(HuberLoss, self).__init__()
        self.delta = delta

    def forward(self, y_pred, y_true):
        err = torch.abs(y_pred - y_true)
        huber_err = torch.where(
            err <= self.delta,
            0.5 * (err ** 2),
            self.delta * (err - 0.5 * self.delta)
        )
        return torch.mean(huber_err)


# =============================================================================
# 3. DATASET LOADING PLACEHOLDER FOR .MAT FILES
# =============================================================================

class MatChannelDataset(Dataset):
    """
    PyTorch Dataset placeholder to load .mat channel datasets.
    Set `mat_dir` to the path containing your .mat dataset files.
    """
    def __init__(self, mat_dir="PATH_TO_YOUR_MAT_FILES_DIRECTORY"):
        super(MatChannelDataset, self).__init__()
        self.mat_dir = mat_dir
        self.samples = []
        
        if os.path.exists(self.mat_dir):
            # Find all .mat files in the specified directory
            self.file_list = [
                os.path.join(self.mat_dir, f) for f in os.listdir(self.mat_dir) if f.endswith('.mat')
            ]
            print(f"[MatChannelDataset] Found {len(self.file_list)} .mat files in '{self.mat_dir}'")
        else:
            self.file_list = []
            print(f"[MatChannelDataset] Directory '{self.mat_dir}' not found. Please update `mat_dir` path!")

    def __len__(self):
        # Return arbitrary length if empty for testing, or actual file count
        return len(self.file_list) if len(self.file_list) > 0 else 100

    def __getitem__(self, idx):
        """
        Loads H_LS pilot vector and true H_full grid from .mat file.
        Adjust key names ('H_LS', 'H_true') according to your MATLAB generator.
        """
        if len(self.file_list) > 0:
            filepath = self.file_list[idx % len(self.file_list)]
            
            # Attempt loading with scipy.io first, fallback to h5py (for v7.3 MAT files)
            try:
                data = sio.loadmat(filepath)
                h_ls_raw = data['H_LS']    # Expected shape: (72,) complex or (36, 2) complex
                h_true_raw = data['H_true'] # Expected shape: (14, 72) complex
            except Exception:
                if h5py is not None:
                    with h5py.File(filepath, 'r') as f:
                        h_ls_raw = np.array(f['H_LS'])
                        h_true_raw = np.array(f['H_true'])
                else:
                    raise RuntimeError("Unable to load .mat file. Please install scipy or h5py.")
            
            # Format real and imaginary parts into 2 channels
            # H_LS -> (72, 2)
            h_ls_real_imag = np.stack([np.real(h_ls_raw).flatten(), np.imag(h_ls_raw).flatten()], axis=-1)
            # H_true -> (14, 72, 2)
            h_true_real_imag = np.stack([np.real(h_true_raw), np.imag(h_true_raw)], axis=-1)

            x_tensor = torch.tensor(h_ls_real_imag, dtype=torch.float32)
            y_tensor = torch.tensor(h_true_real_imag, dtype=torch.float32)
            return x_tensor, y_tensor
        else:
            # Generate dummy sample if dataset path is not set yet
            dummy_x = torch.randn(72, 2, dtype=torch.float32)
            dummy_y = torch.randn(14, 72, 2, dtype=torch.float32)
            return dummy_x, dummy_y


# =============================================================================
# 4. MAIN TEST & SUMMARY SCRIPT
# =============================================================================

if __name__ == '__main__':
    print("=" * 70)
    print("        HA02 CHANNEL ESTIMATION MODEL IMPLEMENTATION (PYTORCH)        ")
    print("=" * 70)
    
    # 1. Instantiate Model
    model = HA02Model()
    model.eval()
    
    # 2. Count parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    encoder_params = sum(p.numel() for p in model.encoder.parameters() if p.requires_grad)
    decoder_params = sum(p.numel() for p in model.decoder.parameters() if p.requires_grad)
    
    print(f"\n[Model Summary]")
    print(f" - Encoder (Transformer Stack) Parameters: {encoder_params:,}")
    print(f" - Decoder (Residual Conv) Parameters:     {decoder_params:,}")
    print(f" - Total HA02 Trainable Parameters:        {total_params:,}")
    print(f"   (Reference in paper: ~105,607 parameters)\n")
    
    # 3. Test Forward Pass with Dummy Batch
    batch_size = 8
    # Input shape: (Batch, 72 pilot elements, 2 channels [Real, Imag])
    dummy_input = torch.randn(batch_size, 72, 2)
    
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"[Forward Pass Verification]")
    print(f" - Input Shape:  {list(dummy_input.shape)}  (Batch, Pilots, Real/Imag)")
    print(f" - Output Shape: {list(output.shape)} (Batch, Symbols=14, Subcarriers=72, Real/Imag)")
    
    # 4. Dataset Loader Verification
    # USER: Replace this path with your directory containing .mat files!
    MAT_FILE_DIR = r"C:\Users\AT30890\Hoctap\1_Hprediction\working\H_predict_NTN\Hest_NTN_UDA_clean\Attention_for_estimation\mat_data"
    
    dataset = MatChannelDataset(mat_dir=MAT_FILE_DIR)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    sample_x, sample_y = next(iter(dataloader))
    print(f"\n[DataLoader Check]")
    print(f" - Sample Batch Input Tensor Shape:  {list(sample_x.shape)}")
    print(f" - Sample Batch Target Tensor Shape: {list(sample_y.shape)}")
    print("=" * 70)
