# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Backbone modules.
"""
from collections import OrderedDict

import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict, List

from util.misc import NestedTensor, is_main_process

from .position_encoding import build_position_encoding

from torchvision.transforms import Resize
from models_mae import MaskedAutoencoderViT
from transformers import ViTModel
from models_mae import mae_vit_base_patch16


class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class BackboneBase(nn.Module):

    def __init__(self, backbone: nn.Module, train_backbone: bool, num_channels: int, return_interm_layers: bool):
        super().__init__()
        for name, parameter in backbone.named_parameters():
            if not train_backbone or 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                parameter.requires_grad_(False)
        if return_interm_layers:
            return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
        else:
            return_layers = {'layer4': "0"}
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.num_channels = num_channels

    def forward(self, tensor_list: NestedTensor):
        xs = self.body(tensor_list.tensors)
        out: Dict[str, NestedTensor] = {}
        for name, x in xs.items():
            m = tensor_list.mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            out[name] = NestedTensor(x, mask)
        return out


class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""
    def __init__(self, name: str,
                 train_backbone: bool,
                 return_interm_layers: bool,
                 dilation: bool):
        backbone = getattr(torchvision.models, name)(
            replace_stride_with_dilation=[False, False, dilation],
            pretrained=is_main_process(), norm_layer=FrozenBatchNorm2d)
        num_channels = 512 if name in ('resnet18', 'resnet34') else 2048
        super().__init__(backbone, train_backbone, num_channels, return_interm_layers)


class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)

    def forward(self, tensor_list: NestedTensor):
        xs = self[0](tensor_list)
        out: List[NestedTensor] = []
        pos = []
        for name, x in xs.items():
            out.append(x)
            # position encoding
            pos.append(self[1](x).to(x.tensors.dtype))

        return out, pos


def build_backbone(args):
    position_embedding = build_position_encoding(args)

    if args.backbone == "vit":
        train_backbone = args.lr_backbone > 0
        return_interm_layers = args.masks  # 現在は中間層を返す設定はサポートしない

        # ViTベースバックボーンを構築
        backbone = ViTBackbone(
            model_name="google/vit-base-patch16-224",
            train_backbone=train_backbone,
            num_channels=768
        )
    elif args.backbone == "wvit":
        train_backbone = args.lr_backbone > 0
        return_interm_layers = args.masks  # 現在は中間層を返す設定はサポートしない

        # ViTベースバックボーンを構築
        backbone = ViTBackbone(
            model_name="google/vit-base-patch16-224",
            train_backbone=train_backbone,
            num_channels=768,
            pretrained_weights="./vit_weights/model_weight.pth"
        )
    elif args.backbone == "usemae":
        train_backbone = args.lr_backbone > 0

        # MAEバックボーンを初期化
        checkpoint_path = args.mae_weights_path
        backbone = ViTMAEBackbone(checkpoint_path=checkpoint_path, train_backbone=train_backbone)
    else:
        train_backbone = args.lr_backbone > 0
        return_interm_layers = args.masks
        backbone = Backbone(args.backbone, train_backbone, return_interm_layers, args.dilation)

    model = Joiner(backbone, position_embedding)
    model.num_channels = backbone.num_channels
    return model

class ViTBackbone(nn.Module):
    """ViTをバックボーンとして使用するためのカスタムクラス"""
    def __init__(self, model_name: str = "google/vit-base-patch16-224", train_backbone: bool = True, num_channels: int = 768, pretrained_weights: str = None):
        super().__init__()
        self.vit = ViTModel.from_pretrained(model_name)
        if pretrained_weights:
            state_dict = torch.load(pretrained_weights, map_location="cpu")
            self.vit.load_state_dict(state_dict, strict=False)
        if not train_backbone:
            for param in self.vit.parameters():
                param.requires_grad_(False)
        self.num_channels = num_channels
        self.input_size = (224, 224)  # ViTの期待する入力サイズ
        self.patch_size = 16  # デフォルトのパッチサイズ (ViTベースモデルの場合)

    def forward(self, tensor_list: NestedTensor):
        x = tensor_list.tensors  # [batch_size, 3, H, W]
        mask = tensor_list.mask  # [batch_size, H, W]

        # 1. 入力画像をViTの期待するサイズにリサイズ
        resized_x = F.interpolate(x, size=self.input_size, mode="bilinear", align_corners=False)

        # 2. マスクもリサイズ
        resized_mask = F.interpolate(mask[None].float(), size=self.input_size).to(torch.bool)[0]

        # 3. ViTによる特徴抽出
        vit_output = self.vit(pixel_values=resized_x)
        features = vit_output.last_hidden_state  # [batch_size, num_patches + 1, hidden_size]

        # 4. クラストークンを除外
        features = features[:, 1:, :]  # クラストークンを削除

        # 5. パッチ数から2Dマップの形状を計算
        batch_size, num_patches, hidden_size = features.size()
        h_patches = self.input_size[0] // self.patch_size
        w_patches = self.input_size[1] // self.patch_size

        assert h_patches * w_patches == num_patches, (
            f"パッチ数 ({num_patches}) と計算された形状 ({h_patches}x{w_patches}) が一致しません。"
        )

        # 特徴を2Dマップ形式に変換
        features = features.permute(0, 2, 1).view(batch_size, hidden_size, h_patches, w_patches)

        # 6. マスクも同じ形状にリサイズ
        mask = F.interpolate(resized_mask[None].float(), size=(h_patches, w_patches)).to(torch.bool)[0]

        return {"0": NestedTensor(features, mask)}

class ViTMAEBackbone(nn.Module):
    """MAEのエンコーダをバックボーンとして使用"""
    def __init__(self, checkpoint_path: str, train_backbone: bool = True, num_channels: int = 256):
        super().__init__()
        # MAEモデルのエンコーダを初期化
        self.mae_model = mae_vit_base_patch16()
        self.num_channels = num_channels

        # 重みをロード
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        self.mae_model.load_state_dict(checkpoint["model"], strict=False)

        if not train_backbone:
            for param in self.mae_model.parameters():
                param.requires_grad_(False)

    def forward(self, tensor_list: NestedTensor):
        x = tensor_list.tensors  # [batch_size, 3, H, W]
        mask = tensor_list.mask  # [batch_size, H, W]

        # 1. 入力画像をMAEの期待するサイズにリサイズ
        resized_x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)

        # 2. パッチ埋め込み
        patches = self.mae_model.patch_embed(resized_x)  # [batch_size, num_patches, embed_dim]

        # 3. クラストークンを追加
        cls_tokens = self.mae_model.cls_token.expand(patches.size(0), -1, -1)  # [batch_size, 1, embed_dim]
        patches = torch.cat((cls_tokens, patches), dim=1)  # [batch_size, num_patches + 1, embed_dim]

        # 4. 位置埋め込みを追加
        patches = patches + self.mae_model.pos_embed  # [batch_size, num_patches + 1, embed_dim]

        # 5. エンコーダブロックに通す
        for blk in self.mae_model.blocks:
            patches = blk(patches)

        # 6. クラストークンを除外
        patches = patches[:, 1:, :]  # クラストークンを削除

        # 7. 特徴を2Dマップ形式に変換
        batch_size, num_patches, hidden_size = patches.size()
        h_patches = int(num_patches**0.5)
        w_patches = num_patches // h_patches
        assert h_patches * w_patches == num_patches, "パッチ数が正方形ではありません"

        features = patches.permute(0, 2, 1).view(batch_size, hidden_size, h_patches, w_patches)

        # 8. マスクも同じ形状にリサイズ
        mask = F.interpolate(mask[None].float(), size=(h_patches, w_patches)).to(torch.bool)[0]

        return {"0": NestedTensor(features, mask)}
    
class useMAEBackbone(nn.Module):
    """MAEのエンコーダをバックボーンとして使用"""
    def __init__(self, checkpoint_path: str, train_backbone: bool = True, num_channels: int = 256):
        super().__init__()
        # MAEモデルのエンコーダを初期化
        self.mae_model = mae_vit_base_patch16()
        self.num_channels = num_channels

        # 重みをロード
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        self.mae_model.load_state_dict(checkpoint["model"], strict=False)

        if not train_backbone:
            for param in self.mae_model.parameters():
                param.requires_grad_(False)

    def forward(self, tensor_list: NestedTensor):
        x = tensor_list.tensors  # [batch_size, 3, H_orig, W_orig]
        mask = tensor_list.mask  # [batch_size, H_orig, W_orig]

        # 元画像のサイズを保存
        orig_h, orig_w = x.shape[2], x.shape[3]

        # 1. 入力画像をMAEの期待するサイズにリサイズ
        resized_x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)

        # 2. パッチ埋め込み
        patches = self.mae_model.patch_embed(resized_x)  # [batch_size, num_patches, embed_dim]

        # 3. クラストークンを追加
        cls_tokens = self.mae_model.cls_token.expand(patches.size(0), -1, -1)  # [batch_size, 1, embed_dim]
        patches = torch.cat((cls_tokens, patches), dim=1)  # [batch_size, num_patches + 1, embed_dim]

        # 4. 位置埋め込みを追加
        patches = patches + self.mae_model.pos_embed  # [batch_size, num_patches + 1, embed_dim]

        # 5. エンコーダブロックに通す
        for blk in self.mae_model.blocks:
            patches = blk(patches)

        # 6. クラストークンを除外
        patches = patches[:, 1:, :]  # クラストークンを削除

        # 7. 特徴を2Dマップ形式に変換
        batch_size, num_patches, hidden_size = patches.size()
        h_patches = int(num_patches**0.5)
        w_patches = num_patches // h_patches
        assert h_patches * w_patches == num_patches, "パッチ数が正方形ではありません"

        features = patches.permute(0, 2, 1).view(batch_size, hidden_size, h_patches, w_patches)

        # 8. 特徴をリサイズ前の画像サイズに戻す
        # 224x224 -> H_orig x W_orig
        features = F.interpolate(features, size=(orig_h, orig_w), mode="bilinear", align_corners=False)

        # 9. マスクも同じ形状にリサイズ
        mask = F.interpolate(mask[None].float(), size=(orig_h, orig_w)).to(torch.bool)[0]

        return {"0": NestedTensor(features, mask)}
