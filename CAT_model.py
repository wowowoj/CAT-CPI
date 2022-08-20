import torch
import torch.nn as nn
from functools import partial
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

if torch.cuda.is_available():
    device = torch.device('cuda')
    print('The code uses GPU...')
else:
    device = torch.device('cpu')
    print('The code uses CPU!!!')


def _init_vit_weights(m):
    """
    ViT weight initialization
    :param m: module
    """
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=.01)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)


class Attention(nn.Module):
    def __init__(self,
                 dim,
                 num_heads=8,
                 qkv_bias=False,
                 drop_ratio=0.,
                 ):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(drop_ratio)

        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(drop_ratio)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Mlp(nn.Module):
    def __init__(self,
                 in_features,
                 hidden_features,
                 out_features,
                 act_layer=nn.GELU,
                 drop=0.):
        super(Mlp, self).__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block(nn.Module):
    def __init__(self,
                 dim,
                 mlp_ratio=4.,
                 norm_layer=nn.LayerNorm,
                 drop_ratio=0.
                 ):
        super(Block, self).__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, drop_ratio=drop_ratio)

        self.drop_path = DropPath(drop_ratio) if drop_ratio > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)

        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            out_features=dim,
            drop=drop_ratio
        )

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class CAT(nn.Module):
    def __init__(self,
                 depth=4,  # depth of transformer
                 mlp_ratio=4.0,  # ratio of mlp hidden dim to embedding dim
                 embed_dim=192,  # embedding dimension
                 drop_ratio=0.,  # dropout rate
                 usemlp=1,
                 ):
        super(CAT, self).__init__()

        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.pos_embed = nn.Parameter(torch.zeros(1, 256, 256))
        self.pos_embed2 = nn.Parameter(torch.zeros(1, 256, 256))
        self.pos_drop = nn.Dropout(p=drop_ratio)

        # stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_ratio, depth)]
        self.blocks = nn.Sequential(*[
            Block(dim=embed_dim,
                  mlp_ratio=mlp_ratio,
                  drop_ratio=dpr[i],
                  )
            for i in range(depth)
        ])

        depth2 = 1
        dpr2 = [x.item() for x in torch.linspace(0, drop_ratio, depth2)]
        self.blocks2 = nn.Sequential(*[
            Block(dim=embed_dim,
                  mlp_ratio=mlp_ratio,
                  drop_ratio=dpr2[i],
                  )
            for i in range(depth2)
        ])

        self.norm = norm_layer(embed_dim)

        if usemlp == 1:
            self.mlp = nn.Identity()
        else:
            self.mlp = Mlp(
                in_features=992,
                hidden_features=2048,
                out_features=992,
                drop=0.2
            )

        self.apply(_init_vit_weights)

        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.02, inplace=True),

            nn.Conv2d(64, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.02, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.conv2d = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=128, kernel_size=3, stride=2, padding=1),  # 128*128*128
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.02, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1),  # 256*32*32
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.02, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 256*16*16
        )

        self.conv1d = nn.Sequential(
            nn.Conv1d(in_channels=256, out_channels=128, kernel_size=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(in_channels=128, out_channels=64, kernel_size=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(in_channels=64, out_channels=32, kernel_size=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
        )

        self.fc_interaction = nn.Linear(992, 2)
        self.embeddings = nn.Embedding(23, 256)

    def forward_features(self, x):
        B, _, _, _ = x.shape
        x = self.cnn(x)
        x = x.reshape(B, 256, -1)
        x = self.pos_drop(x + self.pos_embed)
        x = self.blocks(x)
        x = self.norm(x)
        return x

    def forward_features2(self, x):
        x = self.pos_drop(x + self.pos_embed2)
        x = self.blocks2(x)
        x = self.norm(x)
        return x

    def forward(self, inputs):
        compound, protein = inputs[0], inputs[1]
        B, _, _, _ = compound.shape
        compound = compound.to(device)
        protein = protein.to(device)
        cp_feature = self.forward_features(compound)
        cp_feature = cp_feature.reshape(B, 1, 256, 256)
        pt_feature = self.embeddings(protein)
        pt_feature = self.forward_features2(pt_feature)
        pt_feature = pt_feature.reshape(B, 1, 256, 256)
        z = torch.cat((cp_feature, pt_feature), 1)
        z = self.conv2d(z)
        z = z.reshape(B, 256, -1)
        z = self.conv1d(z)
        z = z.reshape(B, -1)
        z = self.mlp(z)
        interaction = self.fc_interaction(z)
        return interaction

    def __call__(self, data, train=True):
        inputs, correct_interaction = data[:-1], data[-1]
        predicted_interaction = self.forward(inputs)
        correct_interaction = torch.squeeze(correct_interaction)
        loss = F.cross_entropy(predicted_interaction, correct_interaction.to(device))
        correct_labels = correct_interaction.to('cpu').data.numpy()
        ys = F.softmax(predicted_interaction, 1).to('cpu').data.numpy()
        predicted_labels = list(map(lambda x: np.argmax(x), ys))
        predicted_scores = list(map(lambda x: x[1], ys))
        return loss, correct_labels, predicted_labels, predicted_scores


class Train_model(object):
    def __init__(self, model, lr, weight_decay):
        self.model = model
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )

    def train(self, dataset):
        loss, _, _, _ = self.model(dataset)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.to('cpu').data.numpy()

    def save_model(self, model, filename):
        torch.save(model, filename)


class Tester(object):
    def __init__(self, model):
        self.model = model

    def test(self, data):
        (loss, correct_labels, predicted_labels,
         predicted_scores) = self.model(data, train=False)
        return loss, correct_labels, predicted_labels, predicted_scores

    def save_AUCs(self, AUCs, file_name):
        with open(file_name, "a+") as f:
            line = "epoch: " + str(AUCs[0]) + \
                   "    Batch: " + str(AUCs[1]) + \
                   "    loss: " + str(AUCs[2]) + \
                   "    AUC: " + str(AUCs[3]) + \
                   "    AUPRC: " + str(AUCs[7]) + \
                   "    Precision: " + str(AUCs[4]) + \
                   "    Recall: " + str(AUCs[5]) + \
                   "    F1: " + str(AUCs[6]) + \
                   "\n"
            f.write(line)
