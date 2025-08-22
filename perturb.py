import torch
import torchaudio
import os
import yaml
from speechbrain.inference.classifiers import EncoderClassifier
from torch_stoi import NegSTOILoss

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


class Perturbation:
    def __init__(self, config):
        """
        Initializes the perturbation generator based on the HiddenSpeaker method.
        """
        pert_config = config['perturbation']
        self.device = torch.device(config['training']['device'] if torch.cuda.is_available() else "cpu")

        print("\n" + "="*50)
        print("Initializing Perturbation Module (HiddenSpeaker Method)...")

        # 1. Load pre-trained ECAPA-TDNN model from SpeechBrain
        print("Loading pre-trained ECAPA-TDNN model from SpeechBrain...")
        self.surrogate_model = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-xvect-voxceleb",
            savedir=os.path.join("/tmp", "pretrained_models_sb") # Cache directory
        ).to(self.device)
        self.surrogate_model.eval()
        print("Surrogate model loaded.")

        # 2. PGD Attack Parameters
        self.epsilon = pert_config['epsilon']
        self.num_iter = pert_config['num_iter']
        self.learning_rate = pert_config['lr']
        self.alpha = pert_config['alpha'] # Weight for unlearnable loss
        self.beta = pert_config['beta']   # Weight for STFT loss
        self.gamma = pert_config['gamma'] # Weight for STOI loss
        self.target_sample_rate = config['audio']['sample_rate']

        print(f"Params: epsilon={self.epsilon}, num_iter={self.num_iter}, lr={self.learning_rate}")
        print(f"Loss weights: alpha={self.alpha}, beta={self.beta}, gamma={self.gamma}")

        # 3. Instantiate Loss Functions
        self.cosine_similarity_loss = torch.nn.CosineSimilarity(dim=1)
        self.stft_loss_fn = STFTLoss().to(self.device)
        self.stoi_loss_fn = NegSTOILoss(sample_rate=self.target_sample_rate).to(self.device)
        print("="*50 + "\n")


    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Applies HiddenSpeaker perturbation to the input audio waveform.
        :param waveform: Input audio waveform tensor, shape (Batch, Time)
        :return: Perturbed audio waveform tensor
        """
        waveform = waveform.to(self.device)
        waveform.requires_grad = False

        # Initialize perturbation delta
        delta = torch.zeros_like(waveform, requires_grad=True)
        optimizer = torch.optim.Adam([delta], lr=self.learning_rate)

        # Calculate the original embedding as the target
        with torch.no_grad():
            target_embedding = self.surrogate_model.encode_batch(waveform)

        # Iterative optimization loop
        for _ in range(self.num_iter):
            optimizer.zero_grad()

            # Add perturbation and clamp to valid audio range [-1, 1]
            perturbed_audio = torch.clamp(waveform + delta, -1.0, 1.0)

            # --- Calculate Hybrid Loss ---
            # a) "Unlearnable" Loss
            perturbed_embedding = self.surrogate_model.encode_batch(perturbed_audio)
            loss_arc_sim = 1.0 - self.cosine_similarity_loss(perturbed_embedding, target_embedding).mean()

            # b) STFT Loss
            loss_stft = self.stft_loss_fn(perturbed_audio.squeeze(0), waveform.squeeze(0))

            # c) STOI Loss
            loss_stoi = self.stoi_loss_fn(perturbed_audio, waveform)

            # d) Total Hybrid Loss
            total_loss = (self.alpha * loss_arc_sim +
                          self.beta * loss_stft +
                          self.gamma * loss_stoi)

            # Backpropagate and update perturbation
            total_loss.backward()
            optimizer.step()

            # Project perturbation back to the epsilon-ball (L-infinity norm)
            delta.data = torch.clamp(delta.data, -self.epsilon, self.epsilon)
            
            # Ensure the perturbed audio remains in the valid range
            clamped_perturbed_audio = torch.clamp(waveform + delta.data, -1.0, 1.0)
            delta.data = clamped_perturbed_audio - waveform.data


        # Return the perturbed audio on the original device
        return (waveform + delta).detach()