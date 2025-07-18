import os
from typing import Union, List
from collections import OrderedDict
from pkg_resources import packaging
import torch
import numpy as np
from AnomalyCLIP_lib.simple_tokenizer import SimpleTokenizer as _Tokenizer
# from open_clip import tokenizer
# simple_tokenizer = tokenizer.SimpleTokenizer()
from copy import deepcopy
import torch.nn as nn
import torch.nn.functional as F


_tokenizer = _Tokenizer()


def tokenize(texts: Union[str, List[str]], context_length: int = 77, truncate: bool = False) -> Union[torch.IntTensor, torch.LongTensor]:
    """
    Returns the tokenized representation of given input string(s)

    Parameters
    ----------
    texts : Union[str, List[str]]
        An input string or a list of input strings to tokenize

    context_length : int
        The context length to use; all CLIP models use 77 as the context length

    truncate: bool
        Whether to truncate the text in case its encoding is longer than the context length

    Returns
    -------
    A two-dimensional tensor containing the resulting tokens, shape = [number of input strings, context_length].
    We return LongTensor when torch version is <1.8.0, since older index_select requires indices to be long.
    """
    if isinstance(texts, str):
        texts = [texts]

    sot_token = _tokenizer.encoder["<|startoftext|>"]
    eot_token = _tokenizer.encoder["<|endoftext|>"]
    all_tokens = [[sot_token] + _tokenizer.encode(text) + [eot_token] for text in texts]
    if packaging.version.parse(torch.__version__) < packaging.version.parse("1.8.0"):
        result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)
    else:
        result = torch.zeros(len(all_tokens), context_length, dtype=torch.int)

    for i, tokens in enumerate(all_tokens):
        if len(tokens) > context_length:
            if truncate:
                tokens = tokens[:context_length]
                tokens[-1] = eot_token
            else:
                raise RuntimeError(f"Input {texts[i]} is too long for context length {context_length}")
        result[i, :len(tokens)] = torch.tensor(tokens)

    return result

def encode_text_with_prompt_ensemble(model, texts, device):
    """Prompt Ensembling.
        - erzeugt viele verschiedene Prompts, z.B. 'a bad photo of a damaged screw', 'a bright photo of a cable without defect'
        - generierte Prompts werden tokenisiert und durch den Text-Encoder verarbeitet
        - einzelne Klassen-Embeddings werden normalisiert und gemittelt
        - dadurch enthält man ein robustes Mittelwert-Embedding für die Klasse "normal" und "abnormal"
        => diese werden später mit dem Bild-Feature verglichen (cosine similarity), um zu sagen: "Ist das Bild normal oder abnormal"?
    """
    prompt_normal = ['{}', 'flawless {}', 'perfect {}', 'unblemished {}', '{} without flaw', '{} without defect', '{} without damage']
    prompt_abnormal = ['damaged {}', 'broken {}', '{} with flaw', '{} with defect', '{} with damage']
    prompt_state = [prompt_normal, prompt_abnormal]
    prompt_templates = ['a bad photo of a {}.', 'a low resolution photo of the {}.', 'a bad photo of the {}.', 'a cropped photo of the {}.', 'a bright photo of a {}.', 'a dark photo of the {}.', 'a photo of my {}.', 'a photo of the cool {}.', 'a close-up photo of a {}.', 'a black and white photo of the {}.', 'a bright photo of the {}.', 'a cropped photo of a {}.', 'a jpeg corrupted photo of a {}.', 'a blurry photo of the {}.', 'a photo of the {}.', 'a good photo of the {}.', 'a photo of one {}.', 'a close-up photo of the {}.', 'a photo of a {}.', 'a low resolution photo of a {}.', 'a photo of a large {}.', 'a blurry photo of a {}.', 'a jpeg corrupted photo of the {}.', 'a good photo of a {}.', 'a photo of the small {}.', 'a photo of the large {}.', 'a black and white photo of a {}.', 'a dark photo of a {}.', 'a photo of a cool {}.', 'a photo of a small {}.', 'there is a {} in the scene.', 'there is the {} in the scene.', 'this is a {} in the scene.', 'this is the {} in the scene.', 'this is one {} in the scene.']

    text_features = []
    for i in range(len(prompt_state)):
        prompted_state = [state.format(texts[0]) for state in prompt_state[i]]
        prompted_sentence = []
        for s in prompted_state:
            for template in prompt_templates:
                prompted_sentence.append(template.format(s))
        prompted_sentence = tokenize(prompted_sentence)
        class_embeddings = model.encode_text(prompted_sentence.to(device))
        class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
        class_embedding = class_embeddings.mean(dim=0)
        class_embedding /= class_embedding.norm() # robustes Mittelwert-Embedding für Klasse "normal" bzw. "abnormal"
        text_features.append(class_embedding)

    text_features = torch.stack(text_features, dim=1).to(device).t() # 2D-Tensor (normal and abnormal embedding stacked)

    return text_features



def _get_clones(module, N):
    return nn.ModuleList([deepcopy(module) for i in range(N)])
class AnomalyCLIP_PromptLearner(nn.Module):
    def __init__(self, clip_model, design_details): # , batch_size
        super().__init__()
        classnames = ["object"]
        self.n_cls = len(classnames)
        self.n_ctx = design_details["Prompt_length"]
        n_ctx_pos = self.n_ctx
        n_ctx_neg = self.n_ctx
        self.text_encoder_n_ctx = design_details["learnabel_text_embedding_length"] 
        ctx_init_pos = ""
        ctx_init_neg = ""
        dtype = clip_model.transformer.get_cast_dtype()

        ctx_dim = clip_model.ln_final.weight.shape[0] # 768

        # self.batch_size = batch_size # needed for CoCoOp principle

        # vis_dim = clip_model.visual.output_dim # for CoCoOp principle
        
        self.classnames = classnames

        self.state_normal_list = [
            "{}",
        ]

        self.state_anomaly_list = [
            "damaged {}",
        ]
        
        normal_num = len(self.state_normal_list)
        anormaly_num = len(self.state_anomaly_list)
        self.normal_num = normal_num
        self.anormaly_num = anormaly_num

        if ctx_init_pos and ctx_init_neg:
            # use given words to initialize context vectors
            ctx_init_pos = ctx_init_pos.replace("_", " ")
            ctx_init_neg = ctx_init_neg.replace("_", " ")
            n_ctx_pos = len(ctx_init_pos.split(" "))
            n_ctx_neg = len(ctx_init_neg.split(" "))
            #初始化text成bpd编码
            prompt_pos = tokenize(ctx_init_pos)
            prompt_neg = tokenize(ctx_init_neg)
            with torch.no_grad():
                #生成相应的text embedding
                embedding_pos = clip_model.token_embedding(prompt_pos).type(dtype)
                embedding_neg = clip_model.token_embedding(prompt_neg).type(dtype)
            #这些是去除出来EOS 和 # CLS, EOS， 获得可学习的textual prompt
            ctx_vectors_pos = embedding_pos[0, 1: 1 + n_ctx_pos, :]
            ctx_vectors_neg = embedding_neg[0, 1: 1 + n_ctx_neg, :]
            prompt_prefix_pos = ctx_init_pos
            prompt_prefix_neg = ctx_init_neg
            if True:
                ctx_vectors_pos_ = []
                ctx_vectors_neg_ = []
                for _ in range(self.n_cls):
                    ctx_vectors_pos_.append(deepcopy(ctx_vectors_pos))
                    ctx_vectors_neg_.append(deepcopy(ctx_vectors_neg))
                ctx_vectors_pos = torch.stack(ctx_vectors_pos_, dim=0)
                ctx_vectors_neg = torch.stack(ctx_vectors_neg_, dim=0)

        else:
            # Random Initialization
            if True:
                print("Initializing class-specific contexts")
                #这里是cls是类的个数，n_ctx_pos代表learnable token的长度，ctx_dim表示prompt的dimension
                ctx_vectors_pos = torch.empty(self.n_cls, self.normal_num, n_ctx_pos, ctx_dim, dtype=dtype) # n_ctx_pos: wie viele lernbare Tokens für jeden Prompt (z.B. 4 - 8)
                ctx_vectors_neg = torch.empty(self.n_cls, self.anormaly_num, n_ctx_neg, ctx_dim, dtype=dtype) # ctx_dim: Länge des lernbaren Context-Vectors
            else:
                print("Initializing a generic context")
                ctx_vectors_pos = torch.empty(n_ctx_pos, ctx_dim, dtype=dtype)
                ctx_vectors_neg = torch.empty(n_ctx_neg, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors_pos, std=0.02)
            nn.init.normal_(ctx_vectors_neg, std=0.02)
            prompt_prefix_pos = " ".join(["X"] * n_ctx_pos)
            prompt_prefix_neg = " ".join(["X"] * n_ctx_neg)
        self.compound_prompts_depth = design_details["learnabel_text_embedding_depth"]
        self.compound_prompts_text = nn.ParameterList([nn.Parameter(torch.empty(self.text_encoder_n_ctx, ctx_dim))
                                                      for _ in range(self.compound_prompts_depth - 1)])
        for single_para in self.compound_prompts_text:
            print("single_para", single_para.shape)
            nn.init.normal_(single_para, std=0.02)

        single_layer = nn.Linear(ctx_dim, 896)
        self.compound_prompt_projections = _get_clones(single_layer, self.compound_prompts_depth - 1)


        self.ctx_pos = nn.Parameter(ctx_vectors_pos)  # to be optimized
        self.ctx_neg = nn.Parameter(ctx_vectors_neg)  # to be optimized

        # print(f'self.ctx_pos.shape: {self.ctx_pos.shape} | self.ctx_neg.shape: {self.ctx_neg.shape}')
        # print(f'self.ctx_pos.unsqueeze(0).shape: {self.ctx_pos.unsqueeze(0).shape}')

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]


        prompts_pos = [prompt_prefix_pos +  " " + template.format(name)+ "." for template in self.state_normal_list for name in classnames] # "XXXXXXXXXXXX {}." 
                                                                                                                                            # -> "XXXXXXXXXXXX object."
        prompts_neg = [prompt_prefix_neg +  " " + template.format(name)+ "." for template in self.state_anomaly_list for name in classnames] # "XXXXXXXXXXXX damaged {}." 
                                                                                                                                             # -> "XXXXXXXXXXXX damaged object."
                                                                                                                                             # 12 X, because n_ctx_pos = 12, n_ctx_neg = 12

        tokenized_prompts_pos = []
        tokenized_prompts_neg = []
     
        for p_pos in prompts_pos:
            tokenized_prompts_pos.append(tokenize(p_pos))
        for p_neg in prompts_neg:
            tokenized_prompts_neg.append(tokenize(p_neg))
        tokenized_prompts_pos = torch.cat(tokenized_prompts_pos).to('cuda') # without .to('cuda')
        tokenized_prompts_neg = torch.cat(tokenized_prompts_neg).to('cuda') # without .to('cuda')
        #生成相应的text embedding
        with torch.no_grad():
            embedding_pos = clip_model.token_embedding(tokenized_prompts_pos).type(dtype) # see 'AnomalyCLIP.py' (row 444): 
                                                                                          # -> self.token_embedding = nn.Embedding(vocab_size, transformer_width)
                                                                                          # vocab_size = 77, transformer_width = 768 (see 'build_model.py')
            embedding_neg = clip_model.token_embedding(tokenized_prompts_neg).type(dtype)
            n, l, d = embedding_pos.shape
            print("embedding_pos", embedding_pos.shape)
            embedding_pos = embedding_pos.reshape(normal_num, self.n_cls, l, d).permute(1, 0, 2, 3) # (1, 1, 77, 768) -> (1, 1, 77, 768)
            embedding_neg = embedding_neg.reshape(anormaly_num, self.n_cls, l, d).permute(1, 0, 2, 3)


        self.register_buffer("token_prefix_pos", embedding_pos[:, :, :1, :] ) # (1, 1, 1, 768)
        self.register_buffer("token_suffix_pos", embedding_pos[:, :,1 + n_ctx_pos:, :]) # (1, 1, 64, 768)
        self.register_buffer("token_prefix_neg", embedding_neg[:,:, :1, :]) # (1, 1, 1, 768)
        self.register_buffer("token_suffix_neg", embedding_neg[:, :, 1 + n_ctx_neg:, :]) # (1, 1, 64, 768)

        n, d = tokenized_prompts_pos.shape
        tokenized_prompts_pos = tokenized_prompts_pos.reshape(normal_num, self.n_cls, d).permute(1, 0, 2)

        n, d = tokenized_prompts_neg.shape
        tokenized_prompts_neg = tokenized_prompts_neg.reshape(anormaly_num, self.n_cls, d).permute(1, 0, 2)

        self.n_ctx_pos = n_ctx_pos
        self.n_ctx_neg = n_ctx_neg
        # tokenized_prompts = torch.cat([tokenized_prompts_pos, tokenized_prompts_neg], dim=0)  # torch.Tensor
        self.register_buffer("tokenized_prompts_pos", tokenized_prompts_pos)
        self.register_buffer("tokenized_prompts_neg", tokenized_prompts_neg)
        print("tokenized_prompts shape", self.tokenized_prompts_pos.shape, self.tokenized_prompts_neg.shape)


        ##################################################
        # Own experiments:
        ##################################################

        # Experiment 1: Normal and abnormal meta-nets (CoCoOp principle)
        """
        # print(f'vis_dim: {vis_dim} | ctx_dim: {ctx_dim}')
        
        self.meta_net_pos = nn.Sequential(OrderedDict([
            ("linear1", nn.Linear(vis_dim, vis_dim // 16)),
            ("relu", nn.ReLU(inplace=True)),
            ("linear2", nn.Linear(vis_dim // 16, ctx_dim))
        ]))

        self.meta_net_neg = nn.Sequential(OrderedDict([
            ("linear1", nn.Linear(vis_dim, vis_dim // 16)),
            ("relu", nn.ReLU(inplace=True)),
            ("linear2", nn.Linear(vis_dim // 16, ctx_dim))
        ]))
        """

        # Experiment 2: Visual feature fusion
        # self.meta_net_pos = VisualFeatureFusion(n_ctx=self.n_ctx, ctx_dim=ctx_dim, hidden_dim=64)
        # self.meta_net_neg = VisualFeatureFusion(n_ctx=self.n_ctx, ctx_dim=ctx_dim, hidden_dim=64)


    def forward(self, cls_id =None): # for the other experiments adjust the function head like this: def forward(self, im_features, patch_features_list, cls_id=None):     
        
        ctx_pos = self.ctx_pos # (1, 1, n_ctx, ctx_dim)
        ctx_neg = self.ctx_neg # (1, 1, n_ctx, ctx_dim)

        prefix_pos = self.token_prefix_pos # all buffered, see self.register_buffer
        prefix_neg = self.token_prefix_neg
        suffix_pos = self.token_suffix_pos
        suffix_neg = self.token_suffix_neg
        
        # print(f'prefix_pos.shape: {prefix_pos.shape} | suffix_pos.shape: {suffix_pos.shape}')

        """
        # Experiment 1:
        bias_pos = self.meta_net(image_features)
        bias_neg = self.meta_net(image_features)

        ctx_pos = ctx_pos + bias_pos
        ctx_neg = ctx_neg + bias_neg
        """
        
        """
        # Experiment 2:
        bias_pos = self.meta_net_pos(im_features, patch_features_list)
        bias_neg = self.meta_net_neg(im_features, patch_features_list)

        print(f'bias_pos.shape: {bias_pos.shape}')

        ctx_pos = ctx_pos + bias_pos
        ctx_neg = ctx_neg + bias_neg

        print(f'ctx_pos.shape: {ctx_pos.shape}')
        """
        
        # prompts_pos und prompts_neg sind Matrizen mit der Shape (1, 1, vocab_size, transformer_width) = (1, 1, 77, 768)
        # siehe helpers.ipynb Visualisierung von prompts_pos und prompts_neg: 
        # 1. Zeile: nicht optimiert -> prefix_pos
        # 2-12. Zeile: optimiert -> ctx_pos
        # ab 13. Zeile: nicht optimiert -> suffix_pos 
        
        prompts_pos = torch.cat(
            [
                # N(the number of template), 1, dim
                prefix_pos,  # (n_cls, 1, dim) # hier: (1, 1, 1, 768)
                ctx_pos,  # (n_cls, n_ctx, dim) # hier: (1, 1, 12, 768)
                suffix_pos,  # (n_cls, *, dim) # hier: (1, 1, 64, 768)
            ],
            dim=2,
        )

        # print(f'prefix_pos.shape: {prefix_pos.shape}')
        # print(f'ctx_pos.shape:{ctx_pos.shape}')
        # print(f'suffix_pos.shape: {suffix_pos.shape}')
        # print(f'prompts_pos.shape: {prompts_pos.shape}')

        prompts_neg = torch.cat(
            [
                prefix_neg,  # (n_cls, 1, dim)
                ctx_neg,  # (n_cls, n_ctx, dim)
                suffix_neg,  # (n_cls, *, dim)
            ],
            dim=2,
        )

        _, _, l, d = prompts_pos.shape
        prompts_pos = prompts_pos.reshape(-1, l, d)
        _, _, l, d = prompts_neg.shape
        prompts_neg = prompts_neg.reshape(-1, l, d)
        prompts = torch.cat([prompts_pos, prompts_neg], dim=0)


        _, l, d = self.tokenized_prompts_pos.shape # all buffered, see self.register_buffer
        tokenized_prompts_pos = self.tokenized_prompts_pos.reshape(-1,  d)
        _, l, d = self.tokenized_prompts_neg.shape
        tokenized_prompts_neg = self.tokenized_prompts_neg.reshape(-1,  d)
        tokenized_prompts = torch.cat((tokenized_prompts_pos, tokenized_prompts_neg), dim = 0)


        return prompts, tokenized_prompts, self.compound_prompts_text
    

class VisualFeatureFusion(nn.Module):
    """Module to combine global image features (e.g. image_features) and local image features (e.g. patch_features) to refine the learnable text prompts.

    Args:
        - n_ctx: length of learnable word embeddings (12)
        - ctx_dim: embedding length (768)
        - hidden_dim: hidden dimension for mlp, e.g. 64

    Input:
        - im_features: the image features from the vision encoder
        - patch_features: the patch features form the vision encoder

    Returns:
        - features to refine ctx_pos and ctx_neg, respectively
    
    """
    def __init__(self, n_ctx, ctx_dim, hidden_dim): # 12, 768, 64
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d((n_ctx, ctx_dim))
        self.maxp = nn.AdaptiveMaxPool2d((n_ctx, ctx_dim))

        self.proj = nn.Conv2d(1, n_ctx, 1)

        self.mlp = nn.Sequential(
            nn.Conv2d(9, hidden_dim, 3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(hidden_dim, 9, 3, padding=1)
        )

        self.weights = nn.Conv2d(9, 1, 1)

    def forward(self, im_features, patch_features_list):
        # im_features.shape: (8, 768)

        patch_features = torch.stack(patch_features_list, dim=0) # (4, 8, 1297, 768)

        gap_features = self.gap(patch_features) # (4, 8, 12, 768)
        maxp_features = self.maxp(patch_features) # (4, 8, 12, 768)

        im_features = im_features.unsqueeze(0).unsqueeze(0)

        im_features_proj = self.proj(im_features).permute(0, 2, 1, 3) # (1, 8, 12, 768)

        global_features = torch.cat([im_features_proj, gap_features, maxp_features], dim=0).permute(1, 0, 2, 3)

        out = self.mlp(global_features)
        out = self.weights(out)
        out = out.mean(dim=0) # get rid of batch dimension

        return out
    

########################################################################################################################
#
# Some other experiments: 
#   - to try to refine the context in the learnable text prompts
#   - to refine the anomaly regions through an adapter
# 
# As can be read in our documentation, no significant improvements were observed. More detailed experiments required.
#
########################################################################################################################

class GatedPromptModulator(nn.Module):
    def __init__(self, ctx_dim):
        super().__init__()
        self.gate = nn.Linear(ctx_dim, 1)
        self.modulation = nn.Sequential(
            nn.Linear(ctx_dim, ctx_dim // 4),
            nn.ReLU(True),
            nn.Linear(ctx_dim // 4, ctx_dim)
        )

    def forward(self, ctx, im_features):
        B, D = im_features.shape
        _, _, T, D_ = ctx.shape

        gate = F.sigmoid(self.gate(im_features))
        mod = self.modulation(im_features)

        gate = gate.unsqueeze(1).unsqueeze(1)
        mod = mod.unsqueeze(1).unsqueeze(1)

        ctx_expand = ctx.expand(B, -1, -1, -1).clone()
        gate = gate.expand(-1, 1, T, 1)
        mod = mod.expand(-1, 1, T, D)

        modulated_ctx = gate * mod + (1 - gate) * ctx_expand
        modulated_ctx = modulated_ctx.mean(dim=0, keepdim=True) # get rid of batch dimension

        return modulated_ctx
    

class PromptModulatorWithAttention(nn.Module):
    def __init__(self, ctx_dim=768, num_heads=8):
        super().__init__()
        self.patch_weighting = nn.Conv2d(4, 1, 1)

        self.scale = nn.Sequential(
            nn.Linear(ctx_dim, ctx_dim // 4),
            nn.ReLU(True),
            nn.Linear(ctx_dim // 4, ctx_dim)
        )
        self.shift = nn.Sequential(
            nn.Linear(ctx_dim, ctx_dim // 4),
            nn.ReLU(True),
            nn.Linear(ctx_dim // 4, ctx_dim)
        )

        self.cross_attn = nn.MultiheadAttention(embed_dim=ctx_dim, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(ctx_dim)

        self.mlp = nn.Sequential(
            nn.Linear(ctx_dim, ctx_dim // 4),
            nn.ReLU(True),
            nn.Linear(ctx_dim // 4, ctx_dim)
        )

        self.final_norm = nn.LayerNorm(ctx_dim)

    def forward(self, ctx, im_features, patch_features):
        scale = self.scale(im_features).unsqueeze(1).unsqueeze(1)
        shift = self.shift(im_features).unsqueeze(1).unsqueeze(1)

        modulated_ctx = scale * ctx + shift
        modulated_ctx = modulated_ctx.permute(1, 0, 2, 3)

        patch_features = torch.stack(patch_features, dim=0).permute(1, 0, 2, 3)
        patch_features = self.patch_weighting(patch_features).permute(1, 0, 2, 3)

        modulated_ctx = modulated_ctx.squeeze(0)
        patch_features = patch_features.squeeze(0)

        attn, _ = self.cross_attn(modulated_ctx, patch_features, patch_features)
        attn = self.norm(attn + modulated_ctx)

        out = self.mlp(attn)
        out = self.final_norm(out + attn)
        out = out.mean(dim=0, keepdim=True).unsqueeze(0)

        return out
    

class AnomalyRefinementAdapter_old(nn.Module):
    """Adapter between generated anomaly map and clip vision encoder to refine the anomaly map.
    
    Args:
        - hidden_dims (list): hidden channel dimensions which will be used for the encoder decoder

    Informations:
    
        - Encoder-Decoder-Structure (channels): 1 -> hidden_dims[0] -> hidden_dims[1] -> hidden_dims[0] -> 1

    Returns:
        - refined anomaly map (tensor): (3, 512, 512)
    
    """
    def __init__(self, hidden_dims=[32, 64]):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, hidden_dims[0], 3, padding=1),
            nn.BatchNorm2d(hidden_dims[0]),
            nn.ReLU(True),
            nn.Conv2d(hidden_dims[0], hidden_dims[1], 3, padding=1),
            nn.BatchNorm2d(hidden_dims[1]),
            nn.ReLU(True)
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(hidden_dims[1], hidden_dims[0], 3, padding=1),
            nn.BatchNorm2d(hidden_dims[0]),
            nn.ReLU(True),
            nn.Conv2d(hidden_dims[0], 3, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x # (3, 512, 512)
    

class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.ReLU(True),
            nn.Conv2d(in_channels // reduction, in_channels, 1)
        )
        self.act = nn.Sigmoid()

    def forward(self, x):
        w = self.pool(x)
        w = self.mlp(w)
        w = self.act(w)
        out = x * w
        return out
    

class AnomalyRefinementAdapter(nn.Module):
    """Adapter to refine the segmentation map.
    
    Input:
        - Tensor: (4, 2, 512, 512)

    Returns:
        - Tensor: (4, 2, 512, 512)

    Idea:
        - Encoder-Decoder structure with SE-Block:
        (4, 2, 512, 512) -> (4, 64, 512, 512) -> (4, 128, 512, 512) -> SEBlock -> (4, 128, 512, 512) -> (4, 64, 512, 512) -> (4, 2, 512, 512)
    """
    def __init__(self, in_channels, hidden_dims=[64, 128]):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dims[0], 3, padding=1),
            nn.BatchNorm2d(hidden_dims[0]),
            nn.ReLU(True),
            nn.Conv2d(hidden_dims[0], hidden_dims[1], 3, padding=1),
            nn.BatchNorm2d(hidden_dims[1]),
            nn.ReLU(True)
        )

        self.se = SEBlock(hidden_dims[1])

        self.decoder = nn.Sequential(
            nn.Conv2d(hidden_dims[1], hidden_dims[0], 3, padding=1),
            nn.BatchNorm2d(hidden_dims[0]),
            nn.ReLU(True),
            nn.Conv2d(hidden_dims[0], in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(True)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.se(x)
        x = self.decoder(x)
        return x