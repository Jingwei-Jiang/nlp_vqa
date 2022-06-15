import mindspore.nn as nn
import mindspore.ops.functional as F
from mindspore.ops import operations as P

class QuesEmbedding(nn.Cell):
    def __init__(self, input_size=500, output_size=1024, num_layers=1, batch_first=True):
        super(QuesEmbedding, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=output_size, batch_first=batch_first)

    def forward(self, ques):
        # seq_len * N * 500 -> (1 * N * 1024, 1 * N * 1024)
        _, hx = self.lstm(ques)
        # (1 * N * 1024, 1 * N * 1024) -> 1 * N * 1024
        h, _ = hx
        ques_embedding = h[0]
        return ques_embedding

class Attention(nn.Cell):
    def __init__(self, d=1024, k=512, dropout=True):
        super(Attention, self).__init__()
        self.ff_image = nn.Dense(d, k)
        self.ff_ques = nn.Dense(d, k)
        if dropout:
            self.dropout = nn.Dropout(p=0.5)
        self.ff_attention = nn.Dense(k, 1)
        self.expand_dims = P.ExpandDims()

    def forward(self, vi, vq):
        # N * 196 * 1024 -> N * 196 * 512
        hi = self.ff_image(vi)
        # N * 1024 -> N * 512 -> N * 1 * 512
        hq = self.ff_ques(vq)#.ExpandDims(dim=1)
        hq = self.expand_dims(hq, 1)
        # N * 196 * 512
        ha = nn.Tanh(hi + hq)
        if getattr(self, 'dropout'):
            ha = self.dropout(ha)
        # N * 196 * 512 -> N * 196 * 1 -> N * 196
        ha = self.ff_attention(ha).squeeze(axis=2)
        pi = nn.Softmax(ha)
        pi = self.expand_dims(pi, 2)
        # (N * 196 * 1, N * 196 * 1024) -> N * 1024
        vi_attended = (pi * vi).sum(axis=1)
        u = vi_attended + vq
        return u

class SANModel(nn.Cell):
    def __init__(self, vocab_size, word_emb_size=500, emb_size=1024, att_ff_size=512, output_size=1000,
                 num_att_layers=1, num_mlp_layers=1, mode='train', extract_img_features=True, features_dir=None):
        super(SANModel, self).__init__()
        self.mode = mode
        self.features_dir = features_dir
        self.image_channel = ImageEmbedding(output_size=emb_size, mode=mode, extract_img_features=extract_img_features,
                                            features_dir=features_dir)

        self.word_emb_size = word_emb_size
        self.word_embeddings = nn.Embedding(vocab_size, word_emb_size)
        self.ques_channel = QuesEmbedding(
            word_emb_size, output_size=emb_size, num_layers=1, batch_first=False)

        self.san = nn.CellList(
            [Attention(d=emb_size, k=att_ff_size)] * num_att_layers)

        self.mlp = nn.SequentialCell(
            nn.Dropout(p=0.5),
            nn.Dense(emb_size, output_size))

    def forward(self, images, questions, image_ids):
        image_embeddings = self.image_channel(images, image_ids)
        embeds = self.word_embeddings(questions)
        # nbatch = embeds.size()[0]
        # nwords = embeds.size()[1]

        # ques_embeddings = self.ques_channel(embeds.view(nwords, nbatch, self.word_emb_size))
        ques_embeddings = self.ques_channel(embeds)
        vi = image_embeddings
        u = ques_embeddings
        for att_layer in self.san:
            u = att_layer(vi, u)
        output = self.mlp(u)
        return output