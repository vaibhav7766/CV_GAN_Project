import torch
from torch import nn
from torchvision import transforms, models
from torchvision.models import MobileNet_V3_Large_Weights
from transformers import AutoTokenizer
from PIL import Image
import math


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
    MODEL_CHECKPOINT = "best_model_20k.pth"
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

    caption1, caption2, caption3 = generate_caption_for_image(IMAGE_PATH, model, tokenizer, device, MAX_LENGTH)
    print("Generated Caption (top-k):", caption1)
    print("Generated Caption (nucleus):", caption2)
    print("Generated Caption (beam):", caption3)


if __name__ == "__main__":
    inference()
