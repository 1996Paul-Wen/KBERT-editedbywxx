# Author: Wen Xiuxian
# Date: 20201231
'''
note:
    this script gives an example of BiLSTM_CRF model for NER.
    when the data forwards, BiLSTM_CRF Model should take 2 required args(sentence and tags), and 2 args(padding_mask and batch_sequence_max_len) are optional.
    padding_mask is used to generate mask for rnn layer and for crf layer.
    batch_sequence_max_len is used to resize the sequence length of the current batch (batch_size, seq_length) dynamically according to the longest sequence in this batch.
'''

import torch
import torch.nn as nn
import torch.optim as optim
from torchcrf import CRF
import os
import torch.nn.functional as F

# 声明程序可见的gpu核心id
os.environ["CUDA_VISIBLE_DEVICES"] = "5,6"

# 保持每次得到的随机数固定
torch.manual_seed(1)

EMBEDDING_DIM = 5
HIDDEN_DIM = 4
batch_sequence_length = 16

#########################################################
###  Helper functions to make the code more readable. ###
#########################################################

def prepare_sequence(seq, to_ix) -> torch.tensor:
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)

#########################################################
###               Create BiLSTM_CRF model             ###
#########################################################

class BiLSTM_CRF(nn.Module):

    def __init__(self, vocab_size, tag_to_ix, embedding_dim, hidden_dim):
        super(BiLSTM_CRF, self).__init__()
        ### config
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)
        ### layers
        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2, num_layers=1, bidirectional=True)
        # Maps the output of the LSTM into tag space.
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)
        self.crf = CRF(self.tagset_size)

    def forward(self, sentence, tags, padding_mask=None, batch_sequence_max_len=None):
        """

        :param sentence: torch.tensor (batch_size, seq_length)
        :param tags: torch.tensor (batch_size, seq_length)
        :param padding_mask: torch.tensor (batch_size, seq_length) such as [[1,1,1,1,1,0,0,0], [1,1,1,1,1,1,1,1]]  用于求rnn和crf的mask
        :param batch_sequence_max_len: int, refers to the true max length of sentence in the current input batch   用于实现当前batch的动态句子最大长度
        :return: negative_log_likelihood (torch.tensor), predict_tags torch.tensor-(batch_size, batch_sequence_max_len)
        """
        # 避免warning: RNN module weights are not part of single contiguous chunk of memory
        if not hasattr(self, '_flattened'):
            self.lstm.flatten_parameters()
            setattr(self, '_flattened', True)

        # get batch_size of input
        bs = sentence.shape[0]
        # get batch_sequence_max_len
        if batch_sequence_max_len is None or batch_sequence_max_len <= 0:
            batch_sequence_max_len = sentence.shape[1]
        # 基于batch_sequence_max_len动态调整sentence/tags/padding_mask的长度，减少对无用padding的计算
        sentence = sentence[:, :batch_sequence_max_len]
        tags = tags[:, :batch_sequence_max_len]
        if padding_mask is not None:
            padding_mask = padding_mask[:, :batch_sequence_max_len]

        embeds = self.word_embeds(sentence)
        # embeds转置成rnn的输入格式 (seq_length, batch_size, EMBEDDING_DIM)
        embeds = embeds.transpose(0, 1)

        # pack_padded_sequence
        if padding_mask is not None:
            # batch sequence true lengths
            true_lengths = torch.sum(padding_mask, dim=1).to(tags.device)  # batch中每个句子的真实长度，放到对应gpu上
            embeds = nn.utils.rnn.pack_padded_sequence(embeds, true_lengths, batch_first=False, enforce_sorted=False)

        lstm_out, _ = self.lstm(embeds)

        # pad_packed_sequence
        if padding_mask is not None:
            # pad_packed 到 current batch中 真实最长句子的长度，实现以batch为单位的动态pad_packed
            lstm_out, lens_unpacked = nn.utils.rnn.pad_packed_sequence(lstm_out, total_length=batch_sequence_max_len)
            crf_mask = padding_mask.transpose(0, 1).contiguous().byte()  # mask for crf (batch_sequence_max_len, batch_size)
        else:
            crf_mask = None
        # Get the emission scores from the BiLSTM. shape is (seq_length, batch_size, num_tags)
        lstm_feats = self.hidden2tag(lstm_out)

        # 转置tags为(seq_length, batch_size)，适配crf的输入
        tags = tags.transpose(0, 1)
        # Find the best path, and get the negative_log_likelihood loss according to tags
        loss, predict = self.crf(lstm_feats, tags, mask=crf_mask), self.crf.decode(lstm_feats)
        # 取batch的avg loss
        loss = (-loss) / bs
        # list to tensor must be put into the same device
        predict = torch.LongTensor(predict).to(sentence.device)
        # 回正tags
        tags = tags.transpose(0, 1)
        # predict_mask. 将原句子中的padding字符全部预测为tag 0
        if padding_mask is not None:
            predict_mask = padding_mask
        else:
            predict_mask = (tags > 0).long().to(sentence.device)
        predict = predict * predict_mask
        return loss, predict


###################################################################
### define a function to run training demo for BiLSTM_CRF model ###
###################################################################

def runTrainDemo():

    # Make up some training data
    training_data = [(
        "the wall street journal reported today that apple corporation made money".split(),
        "B I I I O O O B I O O".split()
    ), (
        # "georgia tech is a university in georgia all around the world".split(),
        # "B I O O O O B O O O O".split()
        "georgia tech is a university in georgia".split(),
        "B I O O O O B".split()
    )]

    word_to_ix = {'pad': 0}
    for sentence, tags in training_data:
        for word in sentence:
            if word not in word_to_ix:
                word_to_ix[word] = len(word_to_ix)

    tag_to_ix = {'pad': 0, "B": 1, "I": 2, "O": 3}

    model = BiLSTM_CRF(len(word_to_ix), tag_to_ix, EMBEDDING_DIM, HIDDEN_DIM)

    # get device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 数据并行
    if torch.cuda.device_count() > 1:
        print("{} GPUs are available. Let's use them.".format(torch.cuda.device_count()))
        # 在多个GPU上运行您的操作，方法是让模型使用 DataParallel 并行运行
        model = nn.DataParallel(model)
    # 模型加载到device(cpu或gpu)上
    model = model.to(device)
    # # 随机梯度下降优化器
    # optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)
    optimizer = optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-4)

    # prepare a batch of training data
    sentence_in, targets = torch.LongTensor([]), torch.LongTensor([])
    for sentence, tags in training_data:
        # Get our inputs ready for the network, that is,
        # turn them into Tensors of word indices.
        sentence_ = prepare_sequence(sentence, word_to_ix)
        delta = batch_sequence_length - len(sentence_)
        # padding to batch_sequence_length
        sentence_ = F.pad(sentence_, (0, delta), 'constant', word_to_ix['pad'])
        sentence_ = sentence_.contiguous().view(1, len(sentence_))
        sentence_in = torch.cat([sentence_in, sentence_])

        target = torch.tensor([tag_to_ix[t] for t in tags], dtype=torch.long)
        delta = batch_sequence_length - len(target)
        # padding to batch_sequence_length
        target = F.pad(target, (0, delta), 'constant', tag_to_ix['pad'])
        target = target.contiguous().view(1, len(target))
        targets = torch.cat([targets, target])

    padding_mask = (targets > 0).long().to(device)
    sentence_in = sentence_in.to(device)
    targets = targets.to(device)
    batch_true_max_len = int(torch.max(torch.sum(padding_mask, dim=1)).item())

    ### show batch of training data ###
    print('sentence_in')
    print(sentence_in)
    print('targets')
    print(targets)
    print('padding_mask')
    print(padding_mask)

    # Check predictions before training
    with torch.no_grad():
        print('gold tags is ', targets)
        print('predictions before training', model(sentence_in, targets, padding_mask, batch_true_max_len))

    # training
    # Make sure prepare_sequence from earlier in the LSTM section is loaded
    print('-------------start training--------------')
    for epoch in range(100):
        # Step 1. Remember that Pytorch accumulates gradients.
        # We need to clear them out before each instance
        model.zero_grad()

        # Step 2. Run our forward pass.
        loss, predict = model(sentence_in, targets, padding_mask, batch_true_max_len)

        # 多gpu训练一定要加loss = torch.mean(loss)，否则报错grad can be implicitly created only for scalar outputs
        if hasattr(model, "module") and torch.cuda.device_count() > 1:  # model被DataParallel加载会使得model增加attr "module"
            loss = torch.mean(loss)
        print('current loss is ', loss)  # wxx
        print('current predict is ', predict)  # wxx
        print('---------------------')
        # Step 3. Compute the loss, gradients, and update the parameters by
        # calling optimizer.step()
        loss.backward()
        optimizer.step()

    # Check predictions after training
    with torch.no_grad():
        print('gold tags is ', targets)  # wxx
        _, predict = model(sentence_in, targets, padding_mask, batch_true_max_len)
        print('predictions after training', predict)
    # we got close to it!


if __name__ == "__main__":
    runTrainDemo()
