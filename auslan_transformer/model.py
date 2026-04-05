"""
Auslan → English Sign Language Translation Transformer
=======================================================
Input  : Sequence of pose/hand feature vectors  (N, 258)
         258 = 33*4 pose + 21*3 left hand + 21*3 right hand
Output : English token sequence (gloss translation)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Positional Encoding
# ---------------------------------------------------------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 2048):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, d_model)"""
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


# ---------------------------------------------------------------------------
# Feature Projection (continuous → d_model)
# ---------------------------------------------------------------------------
class FeatureProjection(nn.Module):
    """Projects raw 258-d keypoint vectors into model dimension."""

    def __init__(self, input_dim: int = 258, d_model: int = 256, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, 258) → (B, T, d_model)"""
        return self.proj(x)


# ---------------------------------------------------------------------------
# Main Seq2Seq Transformer
# ---------------------------------------------------------------------------
class AuslanTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        input_dim: int = 258,
        d_model: int = 256,
        nhead: int = 8,
        num_encoder_layers: int = 4,
        num_decoder_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        max_seq_len: int = 2048,
        pad_idx: int = 0,
    ):
        super().__init__()
        self.d_model = d_model
        self.pad_idx = pad_idx

        # Encoder side
        self.feature_proj = FeatureProjection(input_dim, d_model, dropout)
        self.enc_pos = PositionalEncoding(d_model, dropout, max_seq_len)

        # Decoder side
        self.token_embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.dec_pos = PositionalEncoding(d_model, dropout, max_seq_len)

        # Transformer core
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,  # Pre-LN for stability
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_encoder_layers, norm=nn.LayerNorm(d_model)
        )

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=num_decoder_layers, norm=nn.LayerNorm(d_model)
        )

        self.output_proj = nn.Linear(d_model, vocab_size)

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        nn.init.normal_(self.token_embed.weight, std=0.02)

    # ------------------------------------------------------------------
    def encode(
        self,
        src: torch.Tensor,
        src_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        src                : (B, T_src, 258)
        src_key_padding_mask: (B, T_src) bool, True = padding
        Returns            : (B, T_src, d_model)
        """
        x = self.feature_proj(src)
        x = self.enc_pos(x)
        return self.encoder(x, src_key_padding_mask=src_key_padding_mask)

    def decode(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: torch.Tensor | None = None,
        tgt_key_padding_mask: torch.Tensor | None = None,
        memory_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        tgt  : (B, T_tgt)  token ids
        memory: (B, T_src, d_model)
        Returns logits (B, T_tgt, vocab_size)
        """
        x = self.token_embed(tgt) * math.sqrt(self.d_model)
        x = self.dec_pos(x)
        out = self.decoder(
            x,
            memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )
        return self.output_proj(out)

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_key_padding_mask: torch.Tensor | None = None,
        tgt_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        src : (B, T_src, 258)
        tgt : (B, T_tgt)   — shifted-right decoder input
        Returns logits (B, T_tgt, vocab_size)
        """
        T_tgt = tgt.size(1)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(
            T_tgt, device=tgt.device
        )
        memory = self.encode(src, src_key_padding_mask)
        return self.decode(
            tgt,
            memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask,
        )

    # ------------------------------------------------------------------
    @torch.no_grad()
    def greedy_decode(
        self,
        src: torch.Tensor,
        src_key_padding_mask: torch.Tensor | None,
        bos_idx: int,
        eos_idx: int,
        max_len: int = 64,
    ) -> list[list[int]]:
        """Greedy inference. Returns list of token-id lists (one per sample)."""
        B = src.size(0)
        memory = self.encode(src, src_key_padding_mask)

        ys = torch.full((B, 1), bos_idx, dtype=torch.long, device=src.device)
        finished = torch.zeros(B, dtype=torch.bool, device=src.device)

        for _ in range(max_len):
            T = ys.size(1)
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(
                T, device=src.device
            )
            logits = self.decode(
                ys, memory, tgt_mask=tgt_mask,
                memory_key_padding_mask=src_key_padding_mask
            )
            next_tok = logits[:, -1].argmax(-1)  # (B,)
            next_tok[finished] = self.pad_idx
            ys = torch.cat([ys, next_tok.unsqueeze(1)], dim=1)
            finished |= next_tok == eos_idx
            if finished.all():
                break

        results = []
        for i in range(B):
            tokens = ys[i, 1:].tolist()
            if eos_idx in tokens:
                tokens = tokens[: tokens.index(eos_idx)]
            results.append(tokens)
        return results

    @torch.no_grad()
    def beam_decode(
        self,
        src: torch.Tensor,
        src_key_padding_mask: torch.Tensor | None,
        bos_idx: int,
        eos_idx: int,
        beam_size: int = 4,
        max_len: int = 64,
        length_penalty: float = 0.6,
    ) -> list[list[int]]:
        """Beam search (single sample at a time for simplicity)."""
        assert src.size(0) == 1, "beam_decode expects batch size 1"
        device = src.device
        memory = self.encode(src, src_key_padding_mask)  # (1, T, d)

        # Expand memory for beam
        memory = memory.expand(beam_size, -1, -1)
        if src_key_padding_mask is not None:
            src_key_padding_mask = src_key_padding_mask.expand(beam_size, -1)

        # beams: list of (score, token_ids)
        beams = [(0.0, [bos_idx])]
        completed = []

        for _ in range(max_len):
            new_beams = []
            for score, tokens in beams:
                if tokens[-1] == eos_idx:
                    completed.append((score, tokens))
                    continue
                ys = torch.tensor(tokens, device=device).unsqueeze(0)
                ys = ys.expand(beam_size, -1)
                T = ys.size(1)
                tgt_mask = nn.Transformer.generate_square_subsequent_mask(T, device=device)
                logits = self.decode(
                    ys, memory, tgt_mask=tgt_mask,
                    memory_key_padding_mask=src_key_padding_mask
                )
                log_probs = F.log_softmax(logits[0, -1], dim=-1)
                topk_lp, topk_idx = log_probs.topk(beam_size)
                for lp, idx in zip(topk_lp.tolist(), topk_idx.tolist()):
                    new_beams.append((score + lp, tokens + [idx]))

            if not new_beams:
                break
            beams = sorted(new_beams, key=lambda x: x[0], reverse=True)[:beam_size]

        completed += beams
        # Apply length penalty and pick best
        best_score, best_tokens = max(
            completed,
            key=lambda x: x[0] / (len(x[1]) ** length_penalty),
        )
        tokens = best_tokens[1:]  # strip BOS
        if eos_idx in tokens:
            tokens = tokens[: tokens.index(eos_idx)]
        return [tokens]