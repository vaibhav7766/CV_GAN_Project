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
from torchvision.models import MobileNet_V3_Large_Weights
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
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225])
        ])

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
class MobileNetV3Encoder(nn.Module):
    def __init__(self):
        super(MobileNetV3Encoder, self).__init__()
        mobilenet = models.mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V1)
        # Use the convolutional features only
        self.features = mobilenet.features
    
    def forward(self, images):
        features = self.features(images)  # shape: (batch, C, H, W)
        batch, C, H, W = features.shape
        # Flatten spatial dimensions: each image becomes a sequence of (H*W) tokens
        features = features.view(batch, C, H * W)  # (batch, C, H*W)
        features = features.transpose(1, 2)        # (batch, H*W, C)
        return features  # e.g., (batch, 49, 960) for a 7x7 feature map

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
            num_image_tokens: Number of spatial tokens from the encoder (e.g., 7x7=49).
            dropout: Dropout rate.
        """
        super(TransformerDecoder, self).__init__()
        self.embed_dim = embed_dim
        self.max_length = max_length
        
        # Embedding layer for target tokens.
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
        # Sinusoidal positional encoding for target tokens.
        self.register_buffer('positional_encoding', self._generate_positional_encoding(max_length, embed_dim))
        
        # Project encoder's spatial features to the decoder embedding space.
        self.feature_proj = nn.Linear(feature_dim, embed_dim)
        
        # Learnable positional embedding for image spatial tokens.
        self.image_pos_embedding = nn.Parameter(torch.randn(1, num_image_tokens, embed_dim))
        
        # Transformer decoder (batch_first=True).
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=hidden_dim, dropout=dropout, batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        # Final linear layer to map decoder outputs to vocabulary logits.
        self.fc_out = nn.Linear(embed_dim, vocab_size)
    
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
        # Project and add learnable positional embedding to the image features.
        memory = self.feature_proj(encoder_features) + self.image_pos_embedding # (batch, num_image_tokens, embed_dim)
       
        # Embed target tokens and add sinusoidal positional encoding.
        tgt_embedded = self.embedding(tgt_input) * math.sqrt(self.embed_dim)
        seq_len = tgt_input.size(1)
        pos_enc = self.positional_encoding[:, :seq_len, :].to(tgt_input.device)
        tgt_embedded = tgt_embedded + pos_enc
        tgt_embedded = self.dropout(tgt_embedded)
        
        # Generate causal mask if not provided.
        if tgt_mask is None:
            tgt_mask = self.generate_square_subsequent_mask(seq_len).to(tgt_input.device)
        
        # Transformer decoder processing.
        decoder_output = self.transformer_decoder(
            tgt_embedded, memory, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask
        )
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

    def _batched_generate_caption(self, images, max_length):
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

    def batched_generate_caption(self, images, max_length, beam_width=3, length_penalty=0.7, repetition_penalty=1.2):
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
                batch_generated = self._batched_generate_caption(images, max_length)
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
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
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


def train():
    # --- Paths, Tokenizer, Dataset, and DataLoader ---
    IMAGE_DIR = "train2017_50k"
    FEATURES_DIR = "train2017_50k_features"
    CAPTIONS_FILE = "merged_captions.json"
    max_length = 30
    batch_size = 32
    tokenizer = AutoTokenizer.from_pretrained("nemotron_tokenizer")
    dataset = ImageCaptionDataset(IMAGE_DIR, CAPTIONS_FILE, tokenizer, max_length=max_length, use_features=True, features_dir=FEATURES_DIR)

    # --- Create Train, Validation, and Test Splits ---
    total_size = len(dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # --- Device Setup ---
    device = torch.device("xpu" if torch.xpu.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

    # --- Hyperparameters ---
    embed_dim = 128
    num_heads = 4
    hidden_dim = 128
    num_layers = 2
    dropout = 0.2
    feature_dim = 960

    # --- Instantiate Encoder, Decoder, and Model ---
    encoder = MobileNetV3Encoder()
    decoder = TransformerDecoder(
        embed_dim=embed_dim,        
        num_heads=num_heads,      
        hidden_dim=hidden_dim,
        vocab_size=tokenizer.vocab_size,
        num_layers=num_layers,    
        max_length=max_length,
        feature_dim=feature_dim,
        dropout=dropout
    )
    model = ImageCaptionModel(encoder, decoder)

    # --- Loss, Optimizer, and Training ---
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=1e-4)

    num_epochs = 50
    scheduler = ReduceLROnPlateau(optimizer)

    model.train()
    model = model.to(device)
    criterion = criterion.to(device)
    model, optimizer = ipex.optimize(model, optimizer=optimizer)

    trainer = ImageCaptionTrainer(model, tokenizer, criterion, optimizer, scheduler, device)
    trainer.train(train_loader, val_loader, num_epochs, patience=10, min_delta=0.001)

    # --- After Training, Evaluate on the Test Set ---
    bleu_score = trainer.evaluate_test_set(test_loader, max_length)
    print("Test BLEU score: {:.4f}".format(bleu_score))

    # --- Load best model then check bleu score ---
    model.load_state_dict(torch.load("best_model.pth", weights_only=True))
    trainer.model = model
    bleu_score = trainer.evaluate_test_set(test_loader, max_length)
    print("Test BLEU score: {:.4f}".format(bleu_score))

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
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
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
    MAX_LENGTH = 30

    embed_dim = 128
    num_heads = 4
    hidden_dim = 128
    num_layers = 2
    dropout = 0.2
    feature_dim = 960
    
    device = "cpu"
    tokenizer = AutoTokenizer.from_pretrained("nemotron_tokenizer")

    encoder = MobileNetV3Encoder()
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

    caption = generate_caption_for_image(IMAGE_PATH, model, tokenizer, device, MAX_LENGTH)
    print("Generated Caption 1 (top-k):", caption[0])
    print("Generated Caption 2 (nucleus):", caption[1])
    print("Generated Caption 3 (beam):", caption[2])


if __name__ == "__main__":
    inference()
