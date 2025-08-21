import torch
import torchaudio
import argparse
import os
from tqdm import tqdm
from speechbrain.inference.classifiers import EncoderClassifier
from torch_stoi import NegSTOILoss

# --- Helper Functions and Classes ---

class STFTLoss(torch.nn.Module):
    """
    STFT Loss module.
    Calculates the L2 distance between the Short-Time Fourier Transform
    of the original and perturbed audio signals.
    """
    def __init__(self, fft_size=1024, hop_size=256, win_length=1024):
        super().__init__()
        self.fft_size = fft_size
        self.hop_size = hop_size
        self.win_length = win_length

    def forward(self, x_perturbed, x_original):
        """
        Args:
            x_perturbed (Tensor): The perturbed audio signal (Batch, Time).
            x_original (Tensor): The original audio signal (Batch, Time).
        Returns:
            Tensor: The STFT loss value.
        """
        stft_original = torch.stft(x_original, 
                                   n_fft=self.fft_size, 
                                   hop_length=self.hop_size, 
                                   win_length=self.win_length, 
                                   window=torch.hann_window(self.win_length, device=x_original.device),
                                   return_complex=False)
        
        stft_perturbed = torch.stft(x_perturbed, 
                                    n_fft=self.fft_size, 
                                    hop_length=self.hop_size, 
                                    win_length=self.win_length,
                                    window=torch.hann_window(self.win_length, device=x_perturbed.device),
                                    return_complex=False)
        
        # Calculate L2 norm (Frobenius norm for matrices) on the magnitude
        return torch.linalg.norm(stft_original - stft_perturbed, 'fro')

def generate_unlearnable_audio(model, audio_path, output_path, config, device):
    """
    Generates an unlearnable audio sample using the HiddenSpeaker method.
    """
    # 1. Load Audio
    try:
        waveform, sample_rate = torchaudio.load(audio_path)
        waveform = waveform.to(device)
    except Exception as e:
        print(f"Error loading {audio_path}: {e}")
        return

    # 2. Preprocess Audio
    # Resample if necessary
    if sample_rate != config['target_sample_rate']:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=config['target_sample_rate']).to(device)
        waveform = resampler(waveform)
    
    # Ensure mono
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
        
    # 3. Initialize Perturbation
    # Start with zero noise, requires_grad=True to allow optimization
    delta = torch.zeros_like(waveform, requires_grad=True).to(device)
    optimizer = torch.optim.Adam([delta], lr=config['learning_rate'])
    
    # Instantiate loss functions
    cosine_similarity_loss = torch.nn.CosineSimilarity(dim=1)
    stft_loss_fn = STFTLoss().to(device)
    # Note: NegSTOILoss calculates 1 - STOI, which is what we need to minimize
    stoi_loss_fn = NegSTOILoss(sample_rate=config['target_sample_rate']).to(device)

    # Calculate the original embedding as the target (we detach it)
    with torch.no_grad():
        target_embedding = model.encode_batch(waveform)

    # 4. Iterative Optimization Loop
    for _ in range(config['iterations']):
        optimizer.zero_grad()

        # Add perturbation and clamp to valid audio range [-1, 1]
        perturbed_audio = torch.clamp(waveform + delta, -1.0, 1.0)
        
        # --- Calculate Hybrid Loss ---
        
        # a) "Unlearnable" Loss (Simulated ArcFace Loss)
        # We push the embedding of the perturbed audio to be identical to the original.
        # This makes the sample "too easy", preventing the model from learning generalizable features.
        perturbed_embedding = model.encode_batch(perturbed_audio)
        # Loss is 1 - similarity. Minimizing this maximizes similarity.
        loss_arc_sim = 1.0 - cosine_similarity_loss(perturbed_embedding, target_embedding).mean()

        # b) STFT Loss (Perceptual Quality in Frequency Domain)
        loss_stft = stft_loss_fn(perturbed_audio.squeeze(0), waveform.squeeze(0))

        # c) STOI Loss (Speech Intelligibility)
        loss_stoi = stoi_loss_fn(perturbed_audio, waveform)

        # d) Total Hybrid Loss
        total_loss = (config['alpha'] * loss_arc_sim + 
                      config['beta'] * loss_stft + 
                      config['gamma'] * loss_stoi)
        
        # Backpropagate and update perturbation
        total_loss.backward()
        optimizer.step()
        
        # Project perturbation back to the epsilon-ball (L-infinity norm)
        delta.data = torch.clamp(delta.data, -config['epsilon'], config['epsilon'])

    # 5. Create and Save Final Audio
    final_perturbed_audio = torch.clamp(waveform + delta.detach(), -1.0, 1.0)
    torchaudio.save(output_path, final_perturbed_audio.cpu(), config['target_sample_rate'])


def main():
    parser = argparse.ArgumentParser(description="Generate Unlearnable Audios using HiddenSpeaker method.")
    parser.add_argument("--input", required=True, help="Path to a single WAV file or a directory of WAV files.")
    parser.add_argument("--output", required=True, help="Path to the output directory.")
    parser.add_argument("--epsilon", type=float, default=0.005, help="Perturbation budget (L-infinity norm).")
    parser.add_argument("--iterations", type=int, default=20, help="Number of optimization iterations per audio file.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for the optimizer.")
    parser.add_argument("--alpha", type=float, default=1.0, help="Weight for the unlearnable loss (L_arc).")
    parser.add_argument("--beta", type=float, default=0.005, help="Weight for the STFT loss.")
    parser.add_argument("--gamma", type=float, default=0.01, help="Weight for the STOI loss.")
    
    args = parser.parse_args()

    # Configuration Dictionary
    config = {
        'epsilon': args.epsilon,
        'iterations': args.iterations,
        'learning_rate': args.lr,
        'alpha': args.alpha,
        'beta': args.beta,
        'gamma': args.gamma,
        'target_sample_rate': 16000
    }

    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load pre-trained model
    print("Loading pre-trained ECAPA-TDNN model from SpeechBrain...")
    # This will download the model automatically on the first run
    model = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir=os.path.join("/tmp", "pretrained_models_sb") # Cache directory
    ).to(device)
    model.eval()
    print("Model loaded.")

    # Create output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)

    # Process files
    if os.path.isfile(args.input):
        files_to_process = [args.input]
    elif os.path.isdir(args.input):
        files_to_process = [os.path.join(args.input, f) for f in os.listdir(args.input) if f.lower().endswith('.wav')]
    else:
        print(f"Error: Input path {args.input} is not a valid file or directory.")
        return

    print(f"Found {len(files_to_process)} audio files to process.")
    for audio_path in tqdm(files_to_process, desc="Processing audio files"):
        base_name = os.path.basename(audio_path)
        output_path = os.path.join(args.output, base_name)
        generate_unlearnable_audio(model, audio_path, output_path, config, device)

    print("Processing complete.")
    print(f"Unlearnable audio files saved in: {args.output}")


if __name__ == "__main__":
    main()
    

# python hiddenspeaker_perturb.py \
#   --input ./librispeech_clean \
#   --output ./librispeech_unlearnable \
#   --epsilon 0.005 \
#   --iterations 20 \
#   --lr 0.001
