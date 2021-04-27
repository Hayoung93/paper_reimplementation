import torch
import torch.nn as nn
from torch import einsum
from einops import rearrange
from itertools import product

from utils import create_mask


class MultiLayerPerceptron(nn.Module):
    def __init__(self, in_dim, hidden_dims, out_dim, activation=None):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dims = hidden_dims
        self.out_dim = out_dim
        if activation is None:
            self.activation = nn.GELU()
        else:
            self.activation = activation
        assert self.hidden_dims is not None, "MLP requires 1 hidden dimension at least"
        if type(self.hidden_dims) == int:
            self.hidden_dims = [self.hidden_dims]
        self.first = nn.Linear(self.in_dim, self.hidden_dims[0])
        self.last = nn.Linear(self.hidden_dims[-1], self.out_dim)
        if len(self.hidden_dims) == 1:
            self.hidden = None
        else:
            self.hidden = nn.ModuleList([])
            for i in range(len(self.hidden_dims) - 1):
                self.hidden.append(nn.Linear(self.hidden_dims[i], self.hidden_dims[i +1]))
    def forward(self, x):
        x = self.first(x)
        if self.hidden is not None:
            for hid in self.hidden:
                x = hid(x)
        x = self.activation(x)
        return self.last(x)


class Residual(nn.Module):
    def __init__(self, anonymousfunction, reshape=None):
        super().__init__()
        self.function = anonymousfunction
        self.reshape = reshape
    
    def forward(self, x):
        out = self.function(x)
        if self.reshape is not None:
            x = self.reshape(x)
        return x + out


class MultiheadSelfAttention(nn.Module):
    def __init__(self, num_head, head_dim, in_dim, shifted, window_size, out_dim):
        super().__init__()
        self.num_head = num_head
        self.head_dim = head_dim
        self.in_dim = in_dim
        self.shifted = shifted
        self.window_size = window_size
        self.out_dim = out_dim
        
        self.pos_embedding = nn.Parameter(torch.randn(2 * self.window_size - 1, 2 * self.window_size - 1))
        self.emb_index = torch.Tensor([list(range(-self.window_size // 2 + 1, -self.window_size // 2 + self.window_size + 1)) for _ in range(self.window_size)])
        self.emb_index = self.emb_index.permute(1, 0) + self.emb_index
        self.emb_index = torch.Tensor(list(product(
            self.emb_index.view(-1), self.emb_index.view(-1)))).view(
                self.window_size ** 2, self.window_size ** 2, 2)
        self.emb_index = self.emb_index + self.window_size - 1
        
        if self.shifted:
            self.mask_horizontal = create_mask(self.window_size, partition_type='h').cuda(1)
            self.mask_vertical = create_mask(self.window_size, partition_type='v').cuda(1)
            # self.mask_cross = create_mask(self.window_size, partition_type='h') + \
            #     create_mask(self.window_size, partition_type='v')
        
        # [B, H, W, C] -> [B, H, W, 3*h*D]
        self.to_qkv = nn.Linear(self.in_dim, self.num_head * self.head_dim * 3)
        self.out_linear = nn.Linear(self.num_head * self.head_dim, self.out_dim)
    
    def forward(self, x):
        B, H, W, C = x.size()
        if H % self.window_size != 0:
            # bottom padding
            self.bottom_pad = nn.ZeroPad2d((0, 0, 0, H % self.window_size))
            x = self.bottom_pad(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        if W % self.window_size != 0:
            # right padding
            self.right_pad = nn.ZeroPad2d(([0, W % self.window_size, 0, 0]))
            x = self.bottom_pad(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        if self.shifted:
            x = torch.roll(x, shifts=(-self.window_size // 2, -self.window_size // 2), dims=(1, 2))
        # three of [B, H, W, h*D]
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        # dot product of q and k at last dimension D: [B nH nW h wHW wHW]
        # where nH/nW: resized patch height/width and wHW: window size
        q, k, v = map(lambda x: \
            rearrange(x, 'B (nH wH) (nW wW) (h D) -> B nH nW h (wH wW) D',
            h=self.num_head, nH=H//self.window_size, wW=self.window_size), qkv)
        # i == nH, j == nW, k == wHW1, l == wHW2
        score = einsum('b i j h k d, b i j h l d -> b i j h k l', q, k)
        # scaling
        score = score / (self.head_dim ** -0.5)
        # relative position embedding
        score += self.pos_embedding[self.emb_index[:, :, 0].long(), self.emb_index[:, :, 1].long()]
        if self.shifted:
            # add mask
            score[:, 0, :, :, :, :] += self.mask_vertical
            score[:, :, 0, :, :, :] += self.mask_horizontal
        # softmax
        score = score.softmax(dim=-1)
        # dot product of score (attention weight) and v (value)
        # i == nH, j == nW, k == wHW1, l == wHW2
        attention = einsum(
            'b i j h k l, b i j h l d -> b i j h k d', score, v)
        # rearrange to [B, H, W, C]
        attention = rearrange(attention,\
            'B nH nW h (wH wW) D -> B (nH wH) (nW wW) (h D)', wH=self.window_size, wW=self.window_size)
        # adjust feature dimension
        output = self.out_linear(attention)
        if self.shifted:
            output = torch.roll(output, shifts=(self.window_size // 2, self.window_size // 2), dims=(1, 2))
        return output


class SwinTransformerBlock(nn.Module):
    def __init__(self, inout_dim, expansion, num_head, head_dim, window_size):
        super().__init__()
        self.inout_dim = inout_dim
        self.expansion = expansion
        self.num_head = num_head
        self.head_dim = head_dim
        self.window_size = window_size

        self.ln_sa1 = Residual(
            nn.Sequential(
                nn.LayerNorm(self.inout_dim),
                MultiheadSelfAttention(self.num_head, self.head_dim, self.inout_dim, False, self.window_size, self.inout_dim)
                ))
        self.ln_mlp1 = Residual(
            nn.Sequential(
                nn.LayerNorm(self.inout_dim),
                MultiLayerPerceptron(self.inout_dim, self.inout_dim * self.expansion, self.inout_dim)
            )
        )
        self.ln_sa2 = Residual(
            nn.Sequential(
                nn.LayerNorm(self.inout_dim),
                MultiheadSelfAttention(self.num_head, self.head_dim, self.inout_dim, True, self.window_size, self.inout_dim)
                ))
        self.ln_mlp2 = Residual(
            nn.Sequential(
                nn.LayerNorm(self.inout_dim),
                MultiLayerPerceptron(self.inout_dim, self.inout_dim * self.expansion, self.inout_dim)
            )
        )
    
    def forward(self, x):
        x = self.ln_sa1(x)
        x = self.ln_mlp1(x)
        x = self.ln_sa2(x)
        return self.ln_mlp2(x)


class PatchEmbed(nn.Module):
    def __init__(self, patch_size, in_dim, out_dim):
        super().__init__()
        self.patch_size = patch_size
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.patching = nn.Unfold(self.patch_size, stride=self.patch_size)
        self.embedding = nn.Linear(self.in_dim * (self.patch_size ** 2), self.out_dim)
    
    def forward(self, x):
        b, c, h, w = x.size()
        # input size: [B, C, H, W] -> patched size: [B, C*PH*PW, L]
        x = self.patching(x)
        x = self.embedding(x.permute(0, 2, 1))
        # output size: [B, L, out_dim] -> [B, newH, newW, out_dim]
        return x.view(b, h // self.patch_size, w // self.patch_size, -1)


class SwinTransformerStage(nn.Module):
    def __init__(self, prefunc, num_block, stage_dim, num_head, head_dim, attn_win_size, expansion=4):
        super().__init__()
        self.prefunc = prefunc
        self.num_block = num_block
        self.stage_dim = stage_dim
        self.num_head = num_head
        self.head_dim = head_dim
        self.attn_win_size = attn_win_size

        self.blocks = nn.ModuleList([])
        for b in range(self.num_block):
            self.blocks.append(SwinTransformerBlock(
                inout_dim=self.stage_dim,
                expansion=4,
                num_head=self.num_head,
                head_dim=self.head_dim,
                window_size=self.attn_win_size
            ))
    
    def forward(self, x):
        x = self.prefunc(x)
        for block in self.blocks:
            x = block(x)
        return x


class SwinTransformer(nn.Module):
    def __init__(self, num_blocks, num_stage_dims, in_dim, num_patching_sizes,
    num_heads, num_head_dims, attn_win_size, expansion=4):
        super().__init__()
        self.num_blocks = num_blocks
        self.in_dim = in_dim
        self.num_stage_dims = num_stage_dims
        self.num_patching_sizes = num_patching_sizes
        self.num_heads = num_heads
        self.num_head_dims = num_head_dims
        self.attn_win_size = attn_win_size
        self.expansion = expansion

        if type(self.num_stage_dims) == int:
            self.num_stage_dims = [self.num_stage_dims * (2 ** i) for i in range(len(num_blocks))]
        if type(self.num_head_dims) == int:
            self.num_head_dims = [self.num_head_dims for _ in range(len(num_blocks))]
        if type(self.num_heads) == int:
            self.num_heads = [self.num_heads * (2 ** i) for _ in range(len(num_blocks))]
        self.num_dims = [self.in_dim]
        self.num_dims.extend(self.num_stage_dims)

        self.stage_modules = nn.ModuleList([])
        self.patchemb_modules = nn.ModuleList([])
        for i in range(len(self.num_blocks)):
            self.patchemb_modules.append(PatchEmbed(self.num_patching_sizes[i], self.num_dims[i], self.num_dims[i + 1]))
            self.stage_modules.append(SwinTransformerStage(
                prefunc=self.patchemb_modules[i],
                num_block=self.num_blocks[i],
                stage_dim=self.num_stage_dims[i],
                num_head=self.num_heads[i],
                head_dim=self.num_head_dims[i],
                attn_win_size=self.attn_win_size
                ))
    
    def forward(self, x):
        for module in self.stage_modules:
            x = module(x)
            x = x.permute(0, 3, 1, 2)
        return x


if __name__ == "__main__":
    net = SwinTransformer(
        num_blocks = (1, 1, 3, 1),
        num_stage_dims = 96,
        in_dim = 3,
        num_patching_sizes = (4, 2, 2, 2),
        num_heads = (3, 6, 12, 24),
        num_head_dims = 32,
        attn_win_size = 7,
        expansion = 4
    )
    dummy_x = torch.randn(1, 3, 224, 224)
    logits = net(dummy_x)  # (1,3)
    # torch.Size([1, 768, 7, 7])
    print(logits.size())
