import os
import csv
import json
import torch
import numpy as np
import intel_extension_for_pytorch as ipex
from torch import nn, optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
from torchvision.models import EfficientNet_B4_Weights
from transformers import AutoTokenizer
from PIL import Image
from tqdm import tqdm
import math
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.meteor_score import meteor_score
from torchmetrics.text import ROUGEScore, BERTScore
from pycocoevalcap.cider.cider import Cider


class ImageCaptionDataset(Dataset):
    def __init__(self, image_dir, captions_file, tokenizer, max_length,
                use_features=False, features_dir=None):
        self.image_dir = image_dir
        with open(captions_file, 'r') as f:
            self.data = json.load(f)
        self.image_filenames = list(self.data.keys())
        self.captions = list(self.data.values())
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.use_features = use_features
        self.features_dir = features_dir
        if features_dir is None:
            self.use_features = False
        
        # Transformation is only used when loading raw images.
        weights = EfficientNet_B4_Weights.IMAGENET1K_V1
        self.transform = weights.transforms()

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        image_filename = self.image_filenames[idx]
        caption = self.captions[idx]
        tokenized = self.tokenizer(
            caption,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        tokenized_caption = {
            "input_ids": tokenized.input_ids.squeeze(), 
            "attention_mask": tokenized.attention_mask.squeeze()
        }
        
        if self.use_features:
            feature_path = os.path.join(self.features_dir, os.path.splitext(image_filename)[0] + ".pt")
            features = torch.load(feature_path, weights_only=True)
            return features, tokenized_caption["input_ids"], tokenized_caption["attention_mask"]
        else:
            image_path = os.path.join(self.image_dir, image_filename)
            image = Image.open(image_path).convert("RGB")
            image = self.transform(image)
            return image, tokenized_caption["input_ids"], tokenized_caption["attention_mask"]

# --- Encoder ---
class EfficientNetEncoder(nn.Module):
    def __init__(self):
        super(EfficientNetEncoder, self).__init__()
        # Load pretrained EfficientNet-B4 model
        efficientnet = models.efficientnet_b4(weights=EfficientNet_B4_Weights.IMAGENET1K_V1)
        # Use the convolutional features (exclude the classification head)
        self.features = efficientnet.features  
        # Optionally, add adaptive pooling to get fixed spatial dimensions
        self.pool = nn.AdaptiveAvgPool2d((7, 7))
        
    def forward(self, images):
        features = self.features(images)  # shape: (batch, C, H, W)
        features = self.pool(features)      # shape: (batch, C, 7, 7)
        batch, C, H, W = features.shape
        # Flatten spatial dimensions: each image becomes a sequence of (H*W) tokens
        features = features.view(batch, C, H * W)  # (batch, C, 49)
        features = features.transpose(1, 2)        # (batch, 49, C)
        return features  # e.g., (batch, 49, feature_dim)


# --- Decoder with Spatial Attention and Teacher Forcing ---
class TransformerDecoder(nn.Module):
    def __init__(self, embed_dim, num_heads, hidden_dim, vocab_size, num_layers, max_length, feature_dim, dropout, num_image_tokens=49):
        """
        Args:
            embed_dim: Embedding dimension for target tokens.
            num_heads: Number of attention heads.
            hidden_dim: Dimension of the feedforward network.
            vocab_size: Size of the target vocabulary.
            num_layers: Number of transformer decoder layers.
            max_length: Maximum length for target sequences.
            feature_dim: Dimension of encoder output channels.
            dropout: Dropout rate.
            num_image_tokens: Number of spatial tokens from the encoder (e.g., 7x7=49).
        """
        super(TransformerDecoder, self).__init__()
        self.embed_dim = embed_dim
        self.max_length = max_length
        
        # Token embedding for target captions.
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('positional_encoding', self._generate_positional_encoding(max_length, embed_dim))
        
        # Project encoder's spatial features to decoder embedding space.
        self.feature_proj = nn.Sequential(
            nn.Linear(feature_dim, embed_dim),
            nn.LayerNorm(embed_dim)  # Normalize features for stability
        )
        
        # Learnable positional embeddings for image tokens.
        self.image_pos_embedding = nn.Parameter(torch.randn(1, num_image_tokens, embed_dim))
        
        # Extra Transformer encoder block for image features
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=hidden_dim, dropout=dropout, batch_first=True)
        self.image_feature_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        
        # Transformer decoder layers.
        decoder_layer = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=hidden_dim, dropout=dropout, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        # Final output projection.
        self.fc_out = nn.Linear(embed_dim, vocab_size)

        # Final layer norm for stability
        self.layer_norm = nn.LayerNorm(embed_dim)


    def _generate_positional_encoding(self, max_len, d_model):
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)  # (1, max_len, d_model)

    def generate_square_subsequent_mask(self, sz):
        """Generates a causal mask (upper-triangular) for target tokens."""
        return torch.triu(torch.ones(sz, sz, dtype=torch.bool), diagonal=1)


    def forward(self, encoder_features, tgt_input, tgt_mask=None, tgt_key_padding_mask=None):
        """
        Args:
            encoder_features: Output from encoder, shape (batch, num_image_tokens, feature_dim).
            tgt_input: Tokenized target sequence (teacher forcing input), shape (batch, tgt_seq_len).
            tgt_mask: (Optional) Causal mask for the target sequence.
            tgt_key_padding_mask: (Optional) Padding mask for target tokens.
        Returns:
            Logits for each target token, shape (batch, tgt_seq_len, vocab_size).
        """
        # Project encoder features and add image positional embeddings.
        memory = self.feature_proj(encoder_features) + self.image_pos_embedding  # (batch, num_image_tokens, embed_dim)
        # Process image features through an extra encoder block.
        memory = self.image_feature_encoder(memory)
        
        # Embed target tokens and add positional encoding.
        tgt_embedded = self.embedding(tgt_input) * math.sqrt(self.embed_dim)
        seq_len = tgt_input.size(1)
        pos_enc = self.positional_encoding[:, :seq_len, :].to(tgt_input.device)
        tgt_embedded = tgt_embedded + pos_enc
        tgt_embedded = self.dropout(tgt_embedded)
        
        # Create causal mask if needed.
        if tgt_mask is None:
            tgt_mask = self.generate_square_subsequent_mask(seq_len).to(tgt_input.device)
        
        # Pass through Transformer decoder.
        decoder_output = self.transformer_decoder(tgt_embedded, memory, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask)
        decoder_output = self.layer_norm(decoder_output)
        logits = self.fc_out(decoder_output)
        return logits


# --- Image Captioning Model ---
class ImageCaptionModel(nn.Module):
    def __init__(self, encoder, decoder, use_features=False):
        super(ImageCaptionModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.use_features = use_features

    def forward(self, x, tgt_input, tgt_mask=None, tgt_key_padding_mask=None):
        # If features are precomputed, x is already the encoder output.
        if self.use_features:
            features = x
        else:
            features = self.encoder(x)
        outputs = self.decoder(features, tgt_input, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask)
        return outputs


# --- Model Trainer ---
class ImageCaptionTrainer:
    def __init__(self, model, tokenizer, criterion, optimizer, scheduler, device,
                 model_save_path="best_model.pth", csv_path="training_results.csv"):
        """
        Args:
            model: The captioning model.
            tokenizer: The tokenizer.
            criterion: Loss function.
            optimizer: Optimizer for training.
            scheduler: Learning rate scheduler.
            device: Device (CPU or GPU).
            model_save_path: Path to save the best model.
            csv_path: Path for CSV logging.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.model_save_path = model_save_path
        self.csv_path = csv_path

        # Initialize metrics
        self.cider = Cider()
        self.meteor = meteor_score
        self.rouge = ROUGEScore()
        self.bert = BERTScore()
        self.bleu = corpus_bleu

    def _teacher_forcing_decode(self, images, input_ids, attention_mask):
        """Decodes using teacher forcing."""
        # Use contiguous to ensure memory layout.
        tgt_input = input_ids[:, :-1].contiguous()
        tgt_target = input_ids[:, 1:].contiguous()
        tgt_key_padding_mask = (attention_mask[:, :-1] == 0).to(self.device)
        outputs = self.model(images, tgt_input, tgt_key_padding_mask=tgt_key_padding_mask)
        return outputs, tgt_target

    def validate(self, dataloader, max_length):
        """Validates the model on a given dataloader."""
        self.model.eval()
        total_val_loss = 0.0
        # all_references = []
        # all_hypotheses = []

        with torch.no_grad():
            progress_bar = tqdm(dataloader, desc="Validation", leave=False)
            for images, input_ids, attention_mask in progress_bar:
                images = images.to(self.device)
                input_ids = input_ids.to(self.device)
                outputs, tgt_target = self._teacher_forcing_decode(images, input_ids, attention_mask)
                loss = self.criterion(outputs.reshape(-1, outputs.size(-1)), tgt_target.reshape(-1))
                total_val_loss += loss.item()

                # Generate predictions using the batched caption generation method.
                # batch_generated = self._batched_generate_caption(images, max_length)
                # for i in range(images.size(0)):
                    # hypothesis = self.tokenizer.decode(batch_generated[i], skip_special_tokens=True)
                    # reference = self.tokenizer.decode(input_ids[i], skip_special_tokens=True)
                    # all_hypotheses.append(hypothesis)
                    # all_references.append(reference)

                progress_bar.set_postfix(val_loss=f"{loss.item():.4f}")
        
        avg_loss = total_val_loss / len(dataloader)

        # # Compute CIDEr.
        # references_dict = {i: [ref] for i, ref in enumerate(all_references)}
        # hypotheses_dict = {i: [hyp] for i, hyp in enumerate(all_hypotheses)}
        # cider_score, _ = self.cider.compute_score(hypotheses_dict, references_dict)
        
        # # Compute METEOR (averaged over samples).
        # meteor_scores = [
        #     self.meteor([ref.split()], hyp.split()) for hyp, ref in zip(all_hypotheses, all_references)
        # ]
        # meteor_score_avg = np.mean(meteor_scores)
        
        # # Compute ROUGE-L.
        # rougel_score = self.rouge(all_hypotheses, all_references)["rougeL_fmeasure"].item()

        # metrics = {
        #     "CIDEr Score": cider_score,
        #     "METEOR Score": meteor_score_avg,
        #     "ROUGE-L Score": rougel_score
        # }    

        return avg_loss

    def train(self, train_loader, val_loader, num_epochs, patience, min_delta, max_length):
        """Training loop with validation, early stopping, and CSV logging."""
        best_loss = float('inf')
        epochs_no_improve = 0

        with open(self.csv_path, "w", newline="") as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(["epoch", "train_loss", "val_loss", "best_val_loss", "lr"])
            # csv_writer.writerow(["epoch", "train_loss", "val_loss", "best_val_loss", "lr", "cider_score", "meteor_score", "rougel_score"])
            csv_file.flush()

            for epoch in range(num_epochs):
                total_loss = 0.0
                self.model.train()
                progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=True)

                for images, input_ids, attention_mask in progress_bar:
                    images = images.to(self.device)
                    input_ids = input_ids.to(self.device)
                    self.optimizer.zero_grad()
                    outputs, tgt_target = self._teacher_forcing_decode(images, input_ids, attention_mask)
                    loss = self.criterion(outputs.reshape(-1, outputs.size(-1)), tgt_target.reshape(-1))
                    loss.backward()
                    self.optimizer.step()

                    total_loss += loss.item()
                    progress_bar.set_postfix(loss=f"{loss.item():.4f}")

                avg_train_loss = total_loss / len(train_loader)
                avg_val_loss = self.validate(val_loader, max_length)

                self.scheduler.step(avg_val_loss)
                current_lr = self.optimizer.param_groups[0]['lr']

                # cider_score = metrics["CIDEr Score"]
                # meteor_score = metrics["METEOR Score"]
                # rougel_score = metrics["ROUGE-L Score"]

                print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | LR: {current_lr:.6f}")
                # print(f"CIDEr: {cider_score:.4f} | METEOR: {meteor_score:.4f} | ROUGE-L: {rougel_score:.4f}")

                # Early stopping and model saving logic.
                if best_loss - avg_val_loss > min_delta:
                    best_loss = avg_val_loss
                    epochs_no_improve = 0
                    torch.save(self.model.state_dict(), self.model_save_path)
                    print(f"--> Best model saved. | Val loss {best_loss:.4f}")
                else:
                    epochs_no_improve += 1

                csv_writer.writerow([epoch+1, avg_train_loss, avg_val_loss, best_loss, current_lr])
                # csv_writer.writerow([epoch+1, avg_train_loss, avg_val_loss, best_loss, current_lr, cider_score, meteor_score, rougel_score])
                csv_file.flush()

                if epochs_no_improve >= patience:
                    print("Early stopping triggered!")
                    break

    def greedy_search_decode(self, images, max_length):
        """Generates captions for a batch of images using greedy decoding."""
        self.model.eval()
        with torch.no_grad():
            images = images.to(self.device)
            if self.model.use_features:
                encoder_features = images
            else:
                encoder_features = self.model.encoder(images)
            batch_size = images.size(0)
            start_token_id = (self.tokenizer.bos_token_id if self.tokenizer.bos_token_id is not None
                              else self.tokenizer.pad_token_id)
            eos_token_id = (self.tokenizer.eos_token_id if self.tokenizer.eos_token_id is not None
                            else self.tokenizer.pad_token_id)

            generated = torch.full((batch_size, 1), start_token_id, dtype=torch.long, device=self.device)
            finished = torch.zeros(batch_size, dtype=torch.bool, device=self.device)

            for _ in range(max_length - 1):
                outputs = self.model.decoder(encoder_features, generated)
                next_token_logits = outputs[:, -1, :]
                next_tokens = next_token_logits.argmax(dim=-1, keepdim=True)
                generated = torch.cat([generated, next_tokens], dim=1)
                finished |= next_tokens.squeeze(1) == eos_token_id
                if finished.all():
                    break
        return generated

    def beam_search_decode(self, encoder_feature, max_length, beam_width=3, length_penalty=0.7, repetition_penalty=1.2):
        """
        Performs beam search decoding for a single image's encoder features.
        
        Args:
            encoder_feature: Tensor of shape (num_image_tokens, feature_dim) for a single image.
            max_length: Maximum length for the generated sequence.
            beam_width: Number of beams to maintain.
            length_penalty: Exponent for length normalization.
            repetition_penalty: Factor to penalize repetition.
        
        Returns:
            best_sequence: List of token ids for the best caption.
        """
        start_token_id = self.tokenizer.bos_token_id if self.tokenizer.bos_token_id is not None else self.tokenizer.pad_token_id
        eos_token_id = self.tokenizer.eos_token_id if self.tokenizer.eos_token_id is not None else self.tokenizer.pad_token_id

        # Each beam is a tuple: (generated_sequence, cumulative_score)
        beams = [([start_token_id], 0.0)]
        
        for _ in range(max_length - 1):
            new_beams = []
            for seq, score in beams:
                # If the beam already ended, keep it as is.
                if seq[-1] == eos_token_id:
                    new_beams.append((seq, score))
                    continue
                
                # Prepare the current sequence.
                seq_tensor = torch.tensor(seq, dtype=torch.long, device=self.device).unsqueeze(0)
                # Unsqueeze encoder_feature to shape (1, num_image_tokens, feature_dim)
                outputs = self.model.decoder(encoder_feature.unsqueeze(0), seq_tensor)
                next_token_logits = outputs[0, -1, :]  # (vocab_size,)
                
                # Apply repetition penalty.
                for token_id in set(seq):
                    next_token_logits[token_id] /= repetition_penalty
                
                log_probs = torch.log_softmax(next_token_logits, dim=-1)
                top_log_probs, top_indices = torch.topk(log_probs, beam_width)
                
                for log_prob, token_id in zip(top_log_probs, top_indices):
                    new_seq = seq + [token_id.item()]
                    # Length normalization: divide cumulative score by (sequence_length ** length_penalty)
                    new_score = (score + log_prob.item()) / (len(new_seq) ** length_penalty)
                    new_beams.append((new_seq, new_score))
            
            # Keep only the best beams.
            beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_width]
            
            # If all beams have produced the end token, break.
            if all(seq[-1] == eos_token_id for seq, _ in beams):
                break

        best_sequence = beams[0][0]
        return best_sequence

    def batched_beam_search_decode(self, images, max_length, beam_width=3, length_penalty=0.7, repetition_penalty=1.2):
        """
        Generates captions for a batch of images using beam search decoding.
        
        Args:
            images: Batch of images.
            max_length: Maximum length for each caption.
            beam_width: Beam width for beam search.
            length_penalty: Exponent for length normalization.
            repetition_penalty: Penalty factor for repeated tokens.
        
        Returns:
            Tensor of generated captions with shape (batch_size, seq_length), padded if necessary.
        """
        self.model.eval()
        generated_sequences = []
        
        with torch.no_grad():
            images = images.to(self.device)
            if self.model.use_features:
                encoder_features = images
            else:
                encoder_features = self.model.encoder(images)
            batch_size = images.size(0)
            
            for i in range(batch_size):
                feature = encoder_features[i]  # (num_image_tokens, feature_dim)
                best_seq = self.beam_search_decode(feature,
                                max_length=max_length,
                                beam_width=beam_width,
                                length_penalty=length_penalty,
                                repetition_penalty=repetition_penalty)
                generated_sequences.append(best_seq)
        
        # Pad sequences to the same length for batch processing.
        max_seq_len = max(len(seq) for seq in generated_sequences)
        padded_sequences = []
        for seq in generated_sequences:
            if len(seq) < max_seq_len:
                seq += [self.tokenizer.pad_token_id] * (max_seq_len - len(seq))
            padded_sequences.append(torch.tensor(seq, dtype=torch.long, device=self.device))
        
        return torch.stack(padded_sequences, dim=0)

    def evaluate_test_set(self, test_loader, max_length):
        """
        Evaluates the model on the test set by generating captions and computing multiple metrics.
        """
        self.model.eval()
        all_references = []
        all_hypotheses = []
        bleu_references = []
        bleu_hypotheses = []
        
        with torch.no_grad():
            for images, input_ids, attention_mask in tqdm(test_loader, desc="Testing"):
                batch_generated = self.batched_beam_search_decode(images, max_length)
                for i in range(images.size(0)):
                    hypothesis = self.tokenizer.decode(batch_generated[i], skip_special_tokens=True)
                    reference = self.tokenizer.decode(input_ids[i], skip_special_tokens=True)
                    all_hypotheses.append(hypothesis)
                    all_references.append(reference)
                    bleu_hypotheses.append(hypothesis.split())
                    bleu_references.append([reference.split()])
        
        # Compute BLEU
        print("Computing BLEU Score...", end=" ")
        bleu_score = self.bleu(bleu_references, bleu_hypotheses)
        print(bleu_score)

        # Compute CIDEr
        print("Computing CIDEr...", end=" ")
        references_dict = {i: [ref] for i, ref in enumerate(all_references)}
        hypotheses_dict = {i: [hyp] for i, hyp in enumerate(all_hypotheses)}
        cider_score, _ = self.cider.compute_score(hypotheses_dict, references_dict)
        print(cider_score)
        
        # Compute METEOR
        print("Computing METEOR...", end=" ")
        meteor_scores = [
            self.meteor([ref.split()], hyp.split()) for hyp, ref in zip(all_hypotheses, all_references)
        ]
        meteor_score = np.mean(meteor_scores)
        print(meteor_score)

        # Compute ROUGE
        print("Computing ROUGE-L Score...", end=" ")
        rougel_score = self.rouge(all_hypotheses, all_references)["rougeL_fmeasure"].item()
        print(rougel_score)

        # Compute BERT
        print("Computing BERT Score...", end=" ")
        bertscore = self.bert(all_hypotheses, all_references)["f1"].mean().item()
        print(bertscore)


        return {
            "BLEU Score": bleu_score,
            "CIDEr Score": cider_score,
            "METEOR Score": meteor_score,
            "ROUGE-L Score": rougel_score,
            "BERT Score": bertscore
        }


def precompute_features(dataset, encoder, device, save_dir, batch_size=32):
    encoder.eval()
    encoder.to(device)
    os.makedirs(save_dir, exist_ok=True)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=True, persistent_workers=True, num_workers=4)
    idx = 0
    with torch.no_grad():
        for images, _, _ in tqdm(loader, desc="Precomputing features"):
            images = images.to(device)
            features = encoder(images)
            for feature in features:
                filename = dataset.image_filenames[idx]
                name, _ = filename.split(".")
                feature_path = os.path.join(save_dir, f"{name}.pt")
                torch.save(feature.cpu(), feature_path)
                idx += 1


# --- Inference function ---
def top_k_sampling_decode(encoder_features, model, tokenizer, device, max_length, k=50):
    start_token_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.pad_token_id
    eos_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.pad_token_id
    
    generated = torch.tensor([start_token_id], device=device).unsqueeze(0)
    model.eval()
    with torch.no_grad():
        for _ in range(max_length - 1):
            outputs = model.decoder(encoder_features, generated)
            next_token_logits = outputs[:, -1, :]
            # Get top-k tokens
            topk_logits, topk_indices = torch.topk(next_token_logits, k, dim=-1)
            probs = torch.softmax(topk_logits, dim=-1)
            # Sample from the top-k tokens
            next_token = topk_indices.gather(-1, torch.multinomial(probs, num_samples=1))
            generated = torch.cat([generated, next_token], dim=1)
            if next_token.item() == eos_token_id:
                break
    caption = tokenizer.decode(generated.squeeze(), skip_special_tokens=True)
    return caption

def nucleus_sampling_decode(encoder_features, model, tokenizer, device, max_length, p=0.9):
    start_token_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.pad_token_id
    eos_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.pad_token_id
    
    generated = torch.tensor([start_token_id], device=device).unsqueeze(0)
    model.eval()
    with torch.no_grad():
        for _ in range(max_length - 1):
            outputs = model.decoder(encoder_features, generated)
            next_token_logits = outputs[:, -1, :]
            probs = torch.softmax(next_token_logits, dim=-1)
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            # Remove tokens with cumulative probability above p
            sorted_indices_to_remove = cumulative_probs > p
            # Ensure at least one token is kept
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            next_token_logits[0, sorted_indices[sorted_indices_to_remove]] = -float('Inf')
            next_token = next_token_logits.argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)
            if next_token.item() == eos_token_id:
                break
    caption = tokenizer.decode(generated.squeeze(), skip_special_tokens=True)
    return caption

def beam_search_decode(encoder_features, model, tokenizer, device, max_length, beam_width=3, length_penalty=0.7, repetition_penalty=1.2):
    start_token_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.pad_token_id
    eos_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.pad_token_id

    # Each beam is a tuple: (generated_sequence, cumulative_score)
    beams = [([start_token_id], 0.0)]
    
    for _ in range(max_length - 1):
        new_beams = []
        for seq, score in beams:
            if seq[-1] == eos_token_id:
                new_beams.append((seq, score))
                continue
            seq_tensor = torch.tensor(seq, dtype=torch.long, device=device).unsqueeze(0)
            outputs = model.decoder(encoder_features, seq_tensor)
            next_token_logits = outputs[0, -1, :]  # (vocab_size,)
            
            # Apply repetition penalty for tokens already generated.
            for token_id in set(seq):
                next_token_logits[token_id] /= repetition_penalty
            
            log_probs = torch.log_softmax(next_token_logits, dim=-1)
            top_log_probs, top_indices = torch.topk(log_probs, beam_width)
            for log_prob, token_id in zip(top_log_probs, top_indices):
                new_seq = seq + [token_id.item()]
                # Normalize the score by length (raise length to a penalty exponent)
                new_score = (score + log_prob.item()) / (len(new_seq) ** length_penalty)
                new_beams.append((new_seq, new_score))
        
        # Keep the best beams
        beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_width]
        if all(seq[-1] == eos_token_id for seq, _ in beams):
            break

    best_sequence = beams[0][0]
    caption = tokenizer.decode(best_sequence, skip_special_tokens=True)
    return caption

def generate_caption_for_image(image_path, model, tokenizer, device, max_length):
    weights = EfficientNet_B4_Weights.IMAGENET1K_V1
    transform = weights.transforms()
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        encoder_features = model.encoder(image)
        caption1 = top_k_sampling_decode(encoder_features, model, tokenizer, device, max_length=max_length)
        caption2 = nucleus_sampling_decode(encoder_features, model, tokenizer, device, max_length=max_length)
        caption3 = beam_search_decode(encoder_features, model, tokenizer, device, max_length=max_length)
    return caption1, caption2, caption3

def inference():
    IMAGE_PATH = input("Enter image path: ")
    MODEL_CHECKPOINT = "best_model.pth"
    MAX_LENGTH = 50

    embed_dim = 256
    num_heads = 8
    hidden_dim = 1024
    num_layers = 4
    dropout = 0.3
    feature_dim = 1792
    
    device = "cpu"
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    tokenizer.bos_token = "[CLS]"
    tokenizer.eos_token = "[SEP]"
    tokenizer.bos_token_id = tokenizer.convert_tokens_to_ids("[CLS]")
    tokenizer.eos_token_id = tokenizer.convert_tokens_to_ids("[SEP]")

    encoder = EfficientNetEncoder()
    decoder = TransformerDecoder(
        embed_dim=embed_dim,
        num_heads=num_heads,
        hidden_dim=hidden_dim,
        vocab_size=tokenizer.vocab_size,
        num_layers=num_layers,
        max_length=MAX_LENGTH,
        feature_dim=feature_dim,
        dropout=dropout
    )
    model = ImageCaptionModel(encoder, decoder)

    model.load_state_dict(torch.load(MODEL_CHECKPOINT, weights_only=True))
    print("Loaded model from", MODEL_CHECKPOINT)

    caption1, caption2, caption3 = generate_caption_for_image(IMAGE_PATH, model, tokenizer, device, MAX_LENGTH)
    print("Generated Caption (top-k):", caption1)
    print("Generated Caption (nucleus):", caption2)
    print("Generated Caption (beam):", caption3)

if __name__ == "__main__":
    inference()
