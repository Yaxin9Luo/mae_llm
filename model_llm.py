import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Config,GPT2Model

from functools import partial

import torch
import torch.nn as nn

from timm.models.vision_transformer import PatchEmbed, Block
from transformers import RobertaModel, RobertaConfig
from util.pos_embed import get_2d_sincos_pos_embed

class PatchEmbedGPTMAE(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=1024, gpt2_model_name='gpt2-medium', norm_pix_loss=False, seed=None):
        super().__init__()
        # Patch Embedding (Encoder)
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        self.num_patches = self.patch_embed.num_patches
        
        # GPT-2 (Decoder)

        self.gpt2_config = GPT2Config.from_pretrained(gpt2_model_name)
        # self.gpt2 = GPT2Model(config=self.gpt2_config)  # This creates a model with random weights
        self.gpt2 = GPT2Model.from_pretrained(gpt2_model_name, config=self.gpt2_config)
        
        # Custom output layer
        self.patch_vocab_size = patch_size**2 * in_chans
        self.output_layer = nn.Linear(self.gpt2_config.n_embd, self.patch_vocab_size)

        # Other necessary components
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim), requires_grad=False)
        self.norm_pix_loss = norm_pix_loss
        self.initialize_weights()
    def check_random_init(self):
        # Check a few weights to see if they're randomly initialized
        print("Checking GPT-2 weights:")
        print(self.gpt2.wte.weight[:5, :5])  # Word embeddings
        print(self.gpt2.h[0].attn.c_attn.weight[:5, :5])  # First layer attention weights
        print(self.gpt2.h[-1].mlp.c_fc.weight[:5, :5])  # Last layer MLP weights

        # You can also check the patch embedding weights
        print("Checking patch embedding weights:")
        print(self.patch_embed.proj.weight[:5, :5, 0, 0])
    def initialize_weights(self):
        # Initialize patch_embed weights
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        
        # Initialize mask_token
        nn.init.normal_(self.output_layer.weight, std=0.02)
        nn.init.zeros_(self.output_layer.bias)
        # Initialize GPT-2 weights
        # self.gpt2.init_weights()
        # Load pre-trained GPT-2 weights
        self.gpt2 = self.gpt2.from_pretrained("gpt2-medium")
    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x
    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore
    
    def forward_encoder(self, x, mask_ratio):
        # Patch embedding
        x = self.patch_embed(x)
        # Masking
        x, mask, ids_restore = self.random_masking(x, mask_ratio) # mask shape is (bs, num_patches) here num_patches is 224 / 16 = 14 * 14
        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        # Use GPT-2 for decoding
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1) # [8, 148, 1024]
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token [8, 196, 1024]
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle [8, 196, 1024]
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token # [8, 197, 1024]
        outputs = self.gpt2(inputs_embeds=x, output_hidden_states=True)
        x = outputs.last_hidden_state
        pred = self.output_layer(x) # [8, 197, 768]
        # remove cls token
        pred = pred[:, 1:, :]
        
        return pred


    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove, 
        """
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward(self, imgs, mask_ratio=0.75):
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)
        loss = self.forward_loss(imgs, pred, mask)
        return loss, pred, mask

# Define model creation functions
def mae_patch_gpt2(**kwargs):
    model = PatchEmbedGPTMAE(**kwargs)
    return model

class PatchEmbedRobertaMAE(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=1024, roberta_model_name='roberta-base', norm_pix_loss=False, seed=None):
        super().__init__()
        # Patch Embedding (Encoder)
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        self.num_patches = self.patch_embed.num_patches
        
        # RoBERTa (Encoder-Decoder)
        self.roberta_config = RobertaConfig.from_pretrained(roberta_model_name)
        self.roberta = RobertaModel.from_pretrained(roberta_model_name, config=self.roberta_config)
        
        # Custom output layer
        self.patch_vocab_size = patch_size**2 * in_chans
        self.output_layer = nn.Linear(self.roberta_config.hidden_size, self.patch_vocab_size)

        # Other necessary components
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim), requires_grad=False)
        self.norm_pix_loss = norm_pix_loss
        self.initialize_weights()
        
    def initialize_weights(self):
        # Initialize patch_embed weights
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        
        # Initialize mask_token
        nn.init.normal_(self.mask_token, std=0.02)
        nn.init.normal_(self.output_layer.weight, std=0.02)
        nn.init.zeros_(self.output_layer.bias)
        
    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x
    
    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore
    
    def forward_encoder(self, x, mask_ratio):
        # Patch embedding
        x = self.patch_embed(x)
        # Masking
        x, mask, ids_restore = self.random_masking(x, mask_ratio)
        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        # Use RoBERTa for decoding
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token
        outputs = self.roberta(inputs_embeds=x, output_hidden_states=True)
        x = outputs.last_hidden_state
        pred = self.output_layer(x)
        # remove cls token
        pred = pred[:, 1:, :]
        
        return pred

    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove, 
        """
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward(self, imgs, mask_ratio=0.75):
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)
        loss = self.forward_loss(imgs, pred, mask)
        return loss, pred, mask

    def extract_features(self, imgs, mask_ratio=0.75):
        latent, _, _ = self.forward_encoder(imgs, mask_ratio)
        return latent
def mae_roberta(**kwargs):
    model = PatchEmbedRobertaMAE(**kwargs)
    return model