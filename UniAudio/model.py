# UniAudio Teams
import base64
import gzip
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, List

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

def attention_mask(loss_mask, prefix_lm=True):
    """
    Generate the attention mask from the loss mask,
    where the loss mask is in the format [Batch, Length].
    Usually, the loss mask would look like:
      <False> ... <True> ... <False>, which represents the
    prefix, the target sequence and padding respectively.

    This function generates the mask for multi-head attention,
    which is in the shape of [Batch, Length, Length] and features:
    (1) the prefix entries can see all each other, if prefix_lm,
        otherwise causal;
    (2) the target entries are causal to each other and can see all
        prefix entries;
    (3) the padding entries can neither been seen nor see all other
        entries.
    """

    # basic preparation
    device = loss_mask.device
    batch_size, q_len = loss_mask.size()
    axis = torch.arange(q_len).to(device)
    # find the start and end time indices of loss duration
    start = axis.unsqueeze(0).masked_fill(~loss_mask, 1e8).min(dim=1).values
    end = axis.unsqueeze(0).masked_fill(~loss_mask, -1e8).max(dim=1).values
    # we strictly require that there is only one continuous True segment
    # for each example in the loss_mask:
    assert torch.all(end - start == loss_mask.int().sum(dim=-1) - 1)

    # (1) make it causal
    mask = (axis.unsqueeze(1) >= axis.unsqueeze(0)).repeat(batch_size, 1, 1)
    # (2) allow non-causaility in prefix part, if prefix_lm
    if prefix_lm:
        mask = torch.where(start.view(batch_size, 1, 1) > axis.view(1, 1, q_len),
                       True, mask)

    # (3) kill the padding
    mask = torch.where(end.view(batch_size, 1, 1) < axis.view(1, 1, q_len),
                       False, mask)

    return mask

@dataclass
class ModelDimensions:
    n_mels: int
    n_audio_ctx: int
    n_audio_state: int
    n_audio_head: int
    n_audio_layer: int
    n_vocab: int
    n_text_ctx: int
    n_text_state: int
    n_text_head: int
    n_text_layer: int


class LayerNorm(nn.LayerNorm):
    def forward(self, x: Tensor) -> Tensor:
        return super().forward(x.float()).type(x.dtype)


class Linear(nn.Linear):
    def forward(self, x: Tensor) -> Tensor:
        return F.linear(
            x,
            self.weight.to(x.dtype),
            None if self.bias is None else self.bias.to(x.dtype),
        )


class Conv1d(nn.Conv1d):
    def _conv_forward(
        self, x: Tensor, weight: Tensor, bias: Optional[Tensor]
    ) -> Tensor:
        return super()._conv_forward(
            x, weight.to(x.dtype), None if bias is None else bias.to(x.dtype)
        )


def sinusoids(length, channels, max_timescale=10000):
    """Returns sinusoids for positional embedding"""
    assert channels % 2 == 0
    log_timescale_increment = np.log(max_timescale) / (channels // 2 - 1)
    inv_timescales = torch.exp(-log_timescale_increment * torch.arange(channels // 2))
    scaled_time = torch.arange(length)[:, np.newaxis] * inv_timescales[np.newaxis, :]
    return torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)


class MultiHeadAttention(nn.Module):
    def __init__(self, n_state: int, n_head: int):
        super().__init__()
        self.n_head = n_head
        self.query = Linear(n_state, n_state)
        self.key = Linear(n_state, n_state, bias=False)
        self.value = Linear(n_state, n_state)
        self.out = Linear(n_state, n_state)

    def forward(
        self,
        x: Tensor,
        xa: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        kv_cache: Optional[dict] = None,
    ):
        q = self.query(x)

        if kv_cache is None or xa is None or self.key not in kv_cache:
            # hooks, if installed (i.e. kv_cache is not None), will prepend the cached kv tensors;
            # otherwise, perform key/value projections for self- or cross-attention as usual.
            k = self.key(x if xa is None else xa)
            v = self.value(x if xa is None else xa)
        else:
            # for cross-attention, calculate keys and values once and reuse in subsequent calls.
            k = kv_cache[self.key]
            v = kv_cache[self.value]

        wv, qk = self.qkv_attention(q, k, v, mask)
        return self.out(wv), qk

    def qkv_attention(
        self, q: Tensor, k: Tensor, v: Tensor, mask: Optional[Tensor] = None
    ):
        n_batch, n_ctx, n_state = q.shape
        scale = (n_state // self.n_head) ** -0.25
        q = q.view(*q.shape[:2], self.n_head, -1).permute(0, 2, 1, 3) * scale
        k = k.view(*k.shape[:2], self.n_head, -1).permute(0, 2, 3, 1) * scale
        v = v.view(*v.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)

        qk = q @ k
        if mask is not None:
            qk = qk + mask[:, :, :n_ctx, :n_ctx]
        qk = qk.float()

        w = F.softmax(qk, dim=-1).to(q.dtype)
        return (w @ v).permute(0, 2, 1, 3).flatten(start_dim=2), qk.detach()


class ResidualAttentionBlock(nn.Module):
    def __init__(self, n_state: int, n_head: int, cross_attention: bool = False):
        super().__init__()

        self.attn = MultiHeadAttention(n_state, n_head)
        self.attn_ln = LayerNorm(n_state)

        self.cross_attn = (
            MultiHeadAttention(n_state, n_head) if cross_attention else None
        )
        self.cross_attn_ln = LayerNorm(n_state) if cross_attention else None

        n_mlp = n_state * 4
        self.mlp = nn.Sequential(
            Linear(n_state, n_mlp), nn.GELU(), Linear(n_mlp, n_state)
        )
        self.mlp_ln = LayerNorm(n_state)

    def forward(
        self,
        x: Tensor,
        xa: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        kv_cache: Optional[dict] = None,
    ):
        x = x + self.attn(self.attn_ln(x), mask=mask, kv_cache=kv_cache)[0]
        if self.cross_attn:
            x = x + self.cross_attn(self.cross_attn_ln(x), xa, kv_cache=kv_cache)[0]
        x = x + self.mlp(self.mlp_ln(x))
        return x


class MegaByteModel(nn.Module):
    def __init__(
        self,
        n_vocab: int,
        n_ctx: int,
        n_state: int,
        n_head: int,
        n_layer: int,
        prefix_lm : bool = True,
        na_repeat: int = 1,
        local_sos: int = 24,
        global_sos: int = 23,
    ):
        super().__init__()
        assert n_state % na_repeat == 0 

        # Global
        self.g_emb = nn.Embedding(n_vocab, n_state // na_repeat)
        self.g_pos = nn.Embedding(n_ctx,   n_state)
        self.g_layers = nn.ModuleList([
            ResidualAttentionBlock(n_state, n_head, cross_attention=False)
            for _ in range(n_layer)
        ])
        self.g_ln = LayerNorm(n_state)

        # Local
        self.l_emb = nn.Embedding(n_vocab, n_state // 1)
        self.l_pos = nn.Embedding(na_repeat,   n_state // 1)
        self.l_layers = nn.ModuleList([
            ResidualAttentionBlock(n_state // 1 , n_head // 1, cross_attention=False)
            for _ in range(min(8, n_layer // 2)) # very urly setting, previous, we set as 8
        ]) # note that we use the max local gpt layer as 8
        self.l_ln = LayerNorm(n_state // 1)

        # Others
        self.g2l_linear = nn.Linear(n_state // na_repeat, n_state // 1)
        self.lm_head = nn.Linear(n_state // 1, n_vocab, bias=False)
        self.continuous_mappings = nn.ModuleDict({
            "text_t5": nn.Linear(768, n_state) # 768: clip output size
        })

        # Params
        self.prefix_lm = prefix_lm
        self.na_repeat = na_repeat
        self.n_ctx = n_ctx
        self.local_sos = local_sos
        self.global_sos = global_sos

        # Mask
        causal_mask = torch.empty(n_ctx, n_ctx).fill_(-np.inf).triu_(1).unsqueeze(0).unsqueeze(0)
        self.register_buffer("causal_mask", causal_mask, persistent=False)

        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * n_layer))
        # TODO: implement prefix_lm

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def add_continuous_segments(self, embs, conti_segs):
        """
        Some input frames are based on continuous vectors. Insert them.
        embs: [B, T, n_state]
        conti_segs: List of List of tuple
        """
        #print('conti_segs ', len(conti_segs), conti_segs)
        if conti_segs is None:
            return embs
        for i in range(embs.size(0)):
            for seg in conti_segs[i]:
                start, end, tp, data = seg
                #print('start, end ', i, start, end, data.shape)
                # + 1 due to global shift
                start, end = start // self.na_repeat + 1, end // self.na_repeat + 1
                data = data[::self.na_repeat]
                data = data.to(embs.device)
                #print('data2 ', data.shape)
                if not tp in self.continuous_mappings:
                    raise ValueError(f"{tp} is not supported as a continuous type")
                data = self.continuous_mappings[tp](data)
                # print('embs ', embs.shape)
                # print('data3 ', data.shape)
                embs[i, start: end] = data

        return embs

    def forward(self,
                x: Tensor,
                mask: Tensor,
                kv_cache: Optional[dict] = None,
                conti_segs: List = None):
        assert x.size() == mask.size()
        assert x.size(1) % self.na_repeat == 0
        assert x.size(1) // self.na_repeat < self.n_ctx
        if self.training:
            assert mask is not None
        """
        This forward function can only process the sequences that have no
        history, a.k.a., empty or none input kv_cache. So call this only 
        when: (1) model training; (2) prefix inference.

        In any case, the local kv_cache is neither needed nor reserved,
        so it's set to None.

        Warning: haven't fully valid prefix_lm = True and the usage of 
        conti_segs
        """
        # for i in x[0]:
        #     print(i)
        # assert 1==2
        B, T = x.size(0), x.size(1) // self.na_repeat
        #print('x0 ', x.shape, mask.shape)
        # Global
        x_global = torch.cat([
            torch.ones_like(x[:, :self.na_repeat]) * self.global_sos,
            x[:, :-self.na_repeat],
            ], dim=-1)
        mask = torch.cat([
            torch.zeros_like(mask[:, :self.na_repeat]).bool(),
            mask[:, :-self.na_repeat],
            ], dim=-1)
        global_logits = self.global_forward(x_global, mask, kv_cache, conti_segs)
        #print('global_logits ', global_logits.shape)
        # Local
        x = x.reshape(B * T, self.na_repeat)
        x_local = torch.cat([
            torch.ones_like(x[:, :1]) * self.local_sos,
            x[:, :-1]
            ], dim=-1)
        # print('x_local ', x_local.shape)
        # assert 1==2
        logits = self.local_forward(x_local, global_logits, None, conti_segs)

        return logits.view(B, T * self.na_repeat, -1)

    def global_forward(self,
                       x: Tensor,
                       mask: Tensor = None, 
                       kv_cache: Optional[dict] = None,
                       conti_segs: List = None):
        B, T = x.size(0), x.size(1) // self.na_repeat
        #print('x ', x.shape)
        if kv_cache is not None and kv_cache != {}:
            # step-wise forward
            prev_len = list(kv_cache.values())[0].size(1)
        else:
            prev_len = 0

        if mask is None:
            attn_mask = None
        elif not self.prefix_lm:
            attn_mask = self.causal_mask
        else:
            # Reverse. True means kill
            attn_mask = ~attention_mask(mask[:, ::self.na_repeat]).unsqueeze(1)
            attn_mask = torch.where(attn_mask, -np.inf, 0.0)
        #print('before add conti')
        x = self.add_continuous_segments(
            self.g_emb(x).view(B, T, -1), conti_segs
        )
        #print('after add conti')
        x = x + self.g_pos.weight[prev_len: prev_len + T].unsqueeze(0)
        for layer in self.g_layers:
            x = layer(x, mask=attn_mask, kv_cache=kv_cache)
        x = self.g_ln(x)

        x = x.view(B * T, self.na_repeat, -1)
        x = self.g2l_linear(x)
        
        return x

    def local_forward(self,
                      x: Tensor,
                      global_logits: Tensor,
                      kv_cache: Optional[dict] = None,
                      conti_segs: List = None):
        B, T = x.size()
       
        if kv_cache is not None and kv_cache != {}:
            prev_len = list(kv_cache.values())[0].size(1)
        else:
            prev_len = 0

        if prev_len == 0:
            attn_mask = self.causal_mask
        else:
            attn_mask = None
        # x = self.add_continuous_segments(self.l_emb(x),
        #     conti_segs, B=B, local=True
        # )
        x = self.l_emb(x) + global_logits + \
            self.l_pos.weight[prev_len: prev_len + T].unsqueeze(0)
        for layer in self.l_layers:
            x = layer(x, mask=attn_mask, kv_cache=kv_cache)
        x = self.l_ln(x) 

        x = self.lm_head(x)

        return x

def CrossEntropyAndAccuracy(logits, y, mask, prefix_lm=True, ignore_id=0):
    y, mask = y.to(logits.device), mask.to(logits.device)
    loss = F.cross_entropy(logits.transpose(1, 2).contiguous(),
                           y.contiguous(),
                           ignore_index=ignore_id, reduction='none')
    pred = logits.argmax(2)

    num_all_tokens = y.ne(ignore_id).int().sum()
    num_tgt_tokens = mask.int().sum()

    acc_all = torch.logical_and(pred.eq(y), y.ne(ignore_id)).int().sum() / num_all_tokens
    acc_tgt = torch.logical_and(pred.eq(y), mask).int().sum() / num_tgt_tokens

    if prefix_lm:
        loss = (loss * mask.int()).sum() / num_tgt_tokens
    else:
        loss = loss.sum() / num_all_tokens

    metrics = {'acc_all': acc_all, 'acc_tgt': acc_tgt, 'loss': loss.clone().detach()}

    return logits, loss, metrics

if __name__ == "__main__":
    torch.manual_seed(888)
    model = MegaByteModel(
            n_vocab = 20,
            n_ctx = 20,
            n_state = 2,
            n_head = 1,
            n_layer = 1,
            prefix_lm = True,
            na_repeat = 2,
            local_sos = 19,
            global_sos= 18,
            )
    seqs =  torch.Tensor([[1,1,2,2,3,4,5,6,0,0], [2,2,3,3,7,8,9,10,11,12]]).long()
    masks = torch.Tensor([[0,0,0,0,1,1,1,1,0,0], [0,0,0,0,1,1,1,1, 1, 1 ]]).bool()
    conti_segs =[
    [(0, 2, 'text_emb', torch.rand(2,2).repeat(1,1))],
    [(2, 4, 'text_emb', torch.rand(2,2).repeat(1,1))]
    ]
    print("seqs and masks: ", seqs, masks)
    print("conti_seqs: ", conti_segs)
    logits = model(seqs, masks, conti_segs=conti_segs)
