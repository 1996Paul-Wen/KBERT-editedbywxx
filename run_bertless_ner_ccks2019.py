# -*- encoding:utf -*-
"""
  This script provides some bertless-model example for NER.
"""
import random
import argparse
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from uer.model_builder import build_model
from uer.utils.config import load_hyperparam
from uer.utils.optimizers import  BertAdam
from uer.utils.constants import *
from uer.utils.vocab import Vocab
from uer.utils.seed import set_seed
from uer.model_saver import save_model
from uer.layers.embeddings import WordEmbedding
import numpy as np
from brain import KnowledgeGraph
from torchcrf import CRF

# # 禁用cudnn
# torch.backends.cudnn.enabled = False

import os
# set visible gpus that can be seen by os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
# os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"
# os.environ["CUDA_VISIBLE_DEVICES"] = "7,6,5,4"


class onlyGru(nn.Module):
    def __init__(self, args):
        """

        :param args: 命令行参数的实例对象
        """
        # config
        super(onlyGru, self).__init__()
        self.hidden_size = args.hidden_size
        self.dropout = args.dropout
        self.tag_to_ix, self.begin_ids = args.labels_map, args.begin_ids
        self.tagset_size = len(self.tag_to_ix)

        # 创建网络结构
        self.embedding = WordEmbedding(args, len(args.vocab))
        self.rnn = nn.GRU(self.hidden_size, self.hidden_size // 2, num_layers=1, bidirectional=True, batch_first=False)
        self.dropout_layer = nn.Dropout(p=self.dropout)
        self.output_layer = nn.Linear(self.hidden_size, self.tagset_size)
        self.softmax = nn.LogSoftmax(dim=-1)  # 在最后一维上sum为1

    def forward(self, src, label, padding_mask=None, batch_sequence_max_len=None):
        """
        Args:
            src: means token_ids  [batch_size x seq_length]
            label: means ner label_ids  [batch_size x seq_length]
            padding_mask: [batch_size x seq_length]
            batch_sequence_max_len: int
        Returns:
            loss: Sequence labeling loss.
            correct: Number of labels that are predicted correctly.
            predict: Predicted label.
            label: Gold label.
        """
        if not hasattr(self, '_flattened'):
            self.rnn.flatten_parameters()
            setattr(self, '_flattened', True)

        # get batch_sequence_max_len
        if batch_sequence_max_len is None or batch_sequence_max_len <= 0:
            batch_sequence_max_len = src.shape[1]
        # reshape输入的数据-基于batch_sequence_max_len动态调整src/label/padding_mask的长度，减少对无用padding的计算
        src = src[:, :batch_sequence_max_len]
        label = label[:, :batch_sequence_max_len]
        if padding_mask is not None:
            padding_mask = padding_mask[:, :batch_sequence_max_len]

        # batch_size
        bs = src.shape[0]

        # Embedding.
        emb = self.embedding(src, None)
        output = emb.transpose(0, 1)  # 转置0维和1维， 当rnn batch_first=False时需要进行转置
        # rnn.pack_padded_sequence
        if padding_mask is not None:
            true_lengths = torch.sum(padding_mask, dim=1).to(label.device)  # batch中每个句子的真实长度，放到对应gpu上
            output = nn.utils.rnn.pack_padded_sequence(output, true_lengths, batch_first=False, enforce_sorted=False)
        output, _ = self.rnn(output)
        # rnn.pad_packed_sequence
        if padding_mask is not None:
            output, lens_unpacked = nn.utils.rnn.pad_packed_sequence(output, total_length=batch_sequence_max_len)
        # dropout
        output = self.dropout_layer(output)
        # mission.
        output = output.transpose(0, 1)
        output = self.output_layer(output)
        # result
        # 通过softmax输出每个token对应各个ner label的概率
        output = self.softmax(output)
        # view(-1, self.tagset_size)指将output转化为 batch token_nums * ner_label_nums 的矩阵
        output = output.contiguous().view(-1, self.tagset_size)

        ###### 拉直data and label，计算loss
        # 将真实label转化为 x * 1的矩阵， x表示token数量，直观上看就是一个label作为新矩阵的一行
        label = label.contiguous().view(-1, 1)
        label_mask = (label > 0).float().to(torch.device(label.device))  # label中的元素大于0则转换为1.0，即非padding字符对应的mask为1
        # one_hot：token_nums * ner_label_nums 的 one hot 矩阵，size与output相同，值为1指示真实标签
        one_hot = torch.zeros(label_mask.size(0), self.tagset_size). \
            to(torch.device(label.device)). \
            scatter_(1, label, 1.0)

        # label smooth
        epsilon = 0.1
        # 平滑后的标签有1-epsilon的概率来自于原分布，有epsilon的概率来自于均匀分布
        label_smooth = (1 - epsilon) * one_hot + epsilon / self.tagset_size
        # * 表示同位置元素相乘
        numerator = -torch.sum(output * label_smooth, 1)
        # label_mask拉长为 1 * x 矩阵， x表示token数量
        label_mask = label_mask.contiguous().view(-1)
        # label拉长为 1 * x 矩阵， x表示token数量
        label = label.contiguous().view(-1)

        numerator = torch.sum(label_mask * numerator)
        loss = numerator
        predict = output.argmax(dim=-1)
        correct = torch.sum(
            label_mask * (predict.eq(label)).float()
        )

        return loss, correct, predict, label

class onlyLstm(nn.Module):
    def __init__(self, args):
        """

        :param args: 命令行参数的实例对象
        """
        # config
        super(onlyLstm, self).__init__()
        self.hidden_size = args.hidden_size
        self.dropout = args.dropout
        self.tag_to_ix, self.begin_ids = args.labels_map, args.begin_ids
        self.tagset_size = len(self.tag_to_ix)

        # 创建网络结构
        self.embedding = WordEmbedding(args, len(args.vocab))
        self.rnn = nn.LSTM(self.hidden_size, self.hidden_size // 2, num_layers=1, bidirectional=True, batch_first=False)
        self.dropout_layer = nn.Dropout(p=self.dropout)
        self.output_layer = nn.Linear(self.hidden_size, self.tagset_size)
        self.softmax = nn.LogSoftmax(dim=-1)  # 在最后一维上sum为1

    def forward(self, src, label, padding_mask=None, batch_sequence_max_len=None):
        """
        Args:
            src: means token_ids  [batch_size x seq_length]
            label: means ner label_ids  [batch_size x seq_length]
            padding_mask: [batch_size x seq_length]
            batch_sequence_max_len: int
        Returns:
            loss: Sequence labeling loss.
            correct: Number of labels that are predicted correctly.
            predict: Predicted label.
            label: Gold label.
        """
        if not hasattr(self, '_flattened'):
            self.rnn.flatten_parameters()
            setattr(self, '_flattened', True)

        # get batch_sequence_max_len
        if batch_sequence_max_len is None or batch_sequence_max_len <= 0:
            batch_sequence_max_len = src.shape[1]
        # reshape输入的数据-基于batch_sequence_max_len动态调整src/label/padding_mask的长度，减少对无用padding的计算
        src = src[:, :batch_sequence_max_len]
        label = label[:, :batch_sequence_max_len]
        if padding_mask is not None:
            padding_mask = padding_mask[:, :batch_sequence_max_len]

        # batch_size
        bs = src.shape[0]

        # Embedding.
        emb = self.embedding(src, None)
        output = emb.transpose(0, 1)  # 转置0维和1维， 当rnn batch_first=False时需要进行转置
        # rnn.pack_padded_sequence
        if padding_mask is not None:
            true_lengths = torch.sum(padding_mask, dim=1).to(label.device)  # batch中每个句子的真实长度，放到对应gpu上
            output = nn.utils.rnn.pack_padded_sequence(output, true_lengths, batch_first=False, enforce_sorted=False)
        output, _ = self.rnn(output)
        # rnn.pad_packed_sequence
        if padding_mask is not None:
            output, lens_unpacked = nn.utils.rnn.pad_packed_sequence(output, total_length=batch_sequence_max_len)
        # dropout
        output = self.dropout_layer(output)
        # mission.
        output = output.transpose(0, 1)
        output = self.output_layer(output)
        # result
        # 通过softmax输出每个token对应各个ner label的概率
        output = self.softmax(output)
        # view(-1, self.tagset_size)指将output转化为 batch token_nums * ner_label_nums 的矩阵
        output = output.contiguous().view(-1, self.tagset_size)

        ###### 拉直data and label，计算loss
        # 将真实label转化为 x * 1的矩阵， x表示token数量，直观上看就是一个label作为新矩阵的一行
        label = label.contiguous().view(-1, 1)
        label_mask = (label > 0).float().to(torch.device(label.device))  # label中的元素大于0则转换为1.0，即非padding字符对应的mask为1
        # one_hot：token_nums * ner_label_nums 的 one hot 矩阵，size与output相同，值为1指示真实标签
        one_hot = torch.zeros(label_mask.size(0), self.tagset_size). \
            to(torch.device(label.device)). \
            scatter_(1, label, 1.0)
        # label smooth
        epsilon = 0.1
        # 平滑后的标签有1-epsilon的概率来自于原分布，有epsilon的概率来自于均匀分布
        label_smooth = (1 - epsilon) * one_hot + epsilon / self.tagset_size

        # * 表示同位置元素相乘
        numerator = -torch.sum(output * label_smooth, 1)
        # label_mask拉长为 1 * x 矩阵， x表示token数量
        label_mask = label_mask.contiguous().view(-1)
        # label拉长为 1 * x 矩阵， x表示token数量
        label = label.contiguous().view(-1)
        # sum  矩阵内所有元素求和，得到分子，表示
        numerator = torch.sum(label_mask * numerator)
        loss = numerator
        predict = output.argmax(dim=-1)
        correct = torch.sum(
            label_mask * (predict.eq(label)).float()
        )

        return loss, correct, predict, label

class gruCrf(nn.Module):
    def __init__(self, args):
        """

        :param args: 命令行参数的实例对象
        """
        # config
        super(gruCrf, self).__init__()
        self.hidden_size = args.hidden_size
        self.dropout = args.dropout
        self.tag_to_ix, self.begin_ids = args.labels_map, args.begin_ids
        self.tagset_size = len(self.tag_to_ix)

        # 创建网络结构
        self.embedding = WordEmbedding(args, len(args.vocab))
        self.rnn = nn.GRU(self.hidden_size, self.hidden_size // 2, num_layers=1, bidirectional=True, batch_first=False)
        self.dropout_layer = nn.Dropout(p=self.dropout)
        self.output_layer = nn.Linear(self.hidden_size, self.tagset_size)
        self.crf = CRF(self.tagset_size, batch_first=False)

    def forward(self, src, label, padding_mask=None, batch_sequence_max_len=None):
        """
        Args:
            src: means token_ids  [batch_size x seq_length]
            label: means ner label_ids  [batch_size x seq_length]
            padding_mask: [batch_size x seq_length]
            batch_sequence_max_len: int
        Returns:
            loss: Sequence labeling loss.
            correct: Number of labels that are predicted correctly.
            predict: Predicted label.
            label: Gold label.
        """
        if not hasattr(self, '_flattened'):
            self.rnn.flatten_parameters()
            setattr(self, '_flattened', True)

        # get batch_sequence_max_len
        if batch_sequence_max_len is None or batch_sequence_max_len <= 0:
            batch_sequence_max_len = src.shape[1]
        # reshape输入的数据-基于batch_sequence_max_len动态调整src/label/padding_mask的长度，减少对无用padding的计算
        src = src[:, :batch_sequence_max_len]
        label = label[:, :batch_sequence_max_len]
        if padding_mask is not None:
            padding_mask = padding_mask[:, :batch_sequence_max_len]

        # batch_size
        bs = src.shape[0]

        # Embedding.
        emb = self.embedding(src, None)
        output = emb.transpose(0, 1)  # 转置0维和1维， 当rnn batch_first=False时需要进行转置
        # rnn.pack_padded_sequence
        if padding_mask is not None:
            true_lengths = torch.sum(padding_mask, dim=1).to(label.device)  # batch中每个句子的真实长度，放到对应gpu上
            output = nn.utils.rnn.pack_padded_sequence(output, true_lengths, batch_first=False, enforce_sorted=False)

        output, _ = self.rnn(output)
        # rnn.pad_packed_sequence
        if padding_mask is not None:
            output, lens_unpacked = nn.utils.rnn.pad_packed_sequence(output, total_length=batch_sequence_max_len)
            crf_mask = padding_mask.transpose(0, 1).contiguous().byte()  # mask for crf (batch_sequence_max_len, batch_size)
        else:
            crf_mask = None

        # dropout
        output = self.dropout_layer(output)
        # mission.
        output = self.output_layer(output)

        # 转置label为(seq_length, batch_size)，适配crf的输入
        label_T = label.transpose(0, 1).contiguous()
        # Find the best path, and get the negative_log_likelihood loss according to tags
        loss, predict = self.crf(output, label_T, mask=crf_mask), self.crf.decode(output)
        # 取batch的avg loss
        loss = (-loss) / bs
        # list to tensor must be put into the same device
        predict = torch.LongTensor(predict).to(label.device)

        ### 计算correct
        # 将真实label转化为 1 * x 的矩阵， x表示token数量，也就是个一维的tensor
        label = label.contiguous().view(-1)
        # 同样拉直predict
        predict = predict.contiguous().view(-1)
        label_mask = (label > 0).float().to(label.device)  # label中的元素大于0则转换为1.0，即非padding字符对应的mask为1, padding对应0
        # view(-1) 表示 label_mask拉长为 1 * x 矩阵， x表示token数量
        label_mask = label_mask.contiguous().view(-1)
        predict = predict * label_mask.long()

        correct = torch.sum((predict.eq(label)).float())

        return loss, correct, predict, label

class lstmCrf(nn.Module):
    def __init__(self, args):
        """

        :param args: 命令行参数的实例对象
        """
        # config
        super(lstmCrf, self).__init__()
        self.hidden_size = args.hidden_size
        self.dropout = args.dropout
        self.tag_to_ix, self.begin_ids = args.labels_map, args.begin_ids
        self.tagset_size = len(self.tag_to_ix)

        # 创建网络结构
        self.embedding = WordEmbedding(args, len(args.vocab))
        self.rnn = nn.LSTM(self.hidden_size, self.hidden_size // 2, num_layers=1, bidirectional=True, batch_first=False)
        self.dropout_layer = nn.Dropout(p=self.dropout)
        self.output_layer = nn.Linear(self.hidden_size, self.tagset_size)
        self.crf = CRF(self.tagset_size, batch_first=False)

    def forward(self, src, label, padding_mask=None, batch_sequence_max_len=None):
        """
        Args:
            src: means token_ids  [batch_size x seq_length]
            label: means ner label_ids  [batch_size x seq_length]
            padding_mask: [batch_size x seq_length]
            batch_sequence_max_len: int
        Returns:
            loss: Sequence labeling loss.
            correct: Number of labels that are predicted correctly.
            predict: Predicted label.
            label: Gold label.
        """
        if not hasattr(self, '_flattened'):
            self.rnn.flatten_parameters()
            setattr(self, '_flattened', True)

        # get batch_sequence_max_len
        if batch_sequence_max_len is None or batch_sequence_max_len <= 0:
            batch_sequence_max_len = src.shape[1]
        # reshape输入的数据-基于batch_sequence_max_len动态调整src/label/padding_mask的长度，减少对无用padding的计算
        src = src[:, :batch_sequence_max_len]
        label = label[:, :batch_sequence_max_len]
        if padding_mask is not None:
            padding_mask = padding_mask[:, :batch_sequence_max_len]

        # batch_size
        bs = src.shape[0]

        # Embedding.
        emb = self.embedding(src, None)
        output = emb.transpose(0, 1)  # 转置0维和1维， 当rnn batch_first=False时需要进行转置
        # rnn.pack_padded_sequence
        if padding_mask is not None:
            true_lengths = torch.sum(padding_mask, dim=1).to(label.device)  # batch中每个句子的真实长度，放到对应gpu上
            output = nn.utils.rnn.pack_padded_sequence(output, true_lengths, batch_first=False, enforce_sorted=False)

        output, _ = self.rnn(output)
        # rnn.pad_packed_sequence
        if padding_mask is not None:
            output, lens_unpacked = nn.utils.rnn.pad_packed_sequence(output, total_length=batch_sequence_max_len)
            crf_mask = padding_mask.transpose(0, 1).contiguous().byte()  # mask for crf (batch_sequence_max_len, batch_size)
        else:
            crf_mask = None
        # dropout
        output = self.dropout_layer(output)
        # mission.
        output = self.output_layer(output)

        # 转置label为(seq_length, batch_size)，适配crf的输入
        label_T = label.transpose(0, 1).contiguous()
        # Find the best path, and get the negative_log_likelihood loss according to tags
        loss, predict = self.crf(output, label_T, mask=crf_mask), self.crf.decode(output)
        # 取batch的avg loss
        loss = (-loss) / bs
        # list to tensor must be put into the same device
        predict = torch.LongTensor(predict).to(label.device)

        ### 计算correct
        # 将真实label转化为 1 * x 的矩阵， x表示token数量，也就是个一维的tensor
        label = label.contiguous().view(-1)
        # 同样拉直predict
        predict = predict.contiguous().view(-1)
        label_mask = (label > 0).float().to(label.device)  # label中的元素大于0则转换为1.0，即非padding字符对应的mask为1, padding对应0
        # view(-1) 表示 label_mask拉长为 1 * x 矩阵， x表示token数量
        label_mask = label_mask.contiguous().view(-1)
        predict = predict * label_mask.long()

        correct = torch.sum((predict.eq(label)).float())

        return loss, correct, predict, label

class onlyCrf(nn.Module):
    def __init__(self, args):
        """

        :param args: 命令行参数的实例对象
        """
        # config
        super(onlyCrf, self).__init__()
        self.hidden_size = args.hidden_size
        self.dropout = args.dropout
        self.tag_to_ix, self.begin_ids = args.labels_map, args.begin_ids
        self.tagset_size = len(self.tag_to_ix)

        # 创建网络结构
        self.embedding = WordEmbedding(args, len(args.vocab))
        self.dropout_layer = nn.Dropout(p=self.dropout)
        self.output_layer = nn.Linear(self.hidden_size, self.tagset_size)
        self.crf = CRF(self.tagset_size, batch_first=False)

    def forward(self, src, label, padding_mask=None, batch_sequence_max_len=None):
        """
        Args:
            src: means token_ids  [batch_size x seq_length]
            label: means ner label_ids  [batch_size x seq_length]
            padding_mask: [batch_size x seq_length]
            batch_sequence_max_len: int
        Returns:
            loss: Sequence labeling loss.
            correct: Number of labels that are predicted correctly.
            predict: Predicted label.
            label: Gold label.
        """
        # get batch_sequence_max_len
        if batch_sequence_max_len is None or batch_sequence_max_len <= 0:
            batch_sequence_max_len = src.shape[1]
        # reshape输入的数据-基于batch_sequence_max_len动态调整src/label/mask/pos/vm/padding_mask的长度，减少对无用padding的计算
        src = src[:, :batch_sequence_max_len]
        label = label[:, :batch_sequence_max_len]
        if padding_mask is not None:
            padding_mask = padding_mask[:, :batch_sequence_max_len]

        # Embedding.
        emb = self.embedding(src, None)
        output = emb
        # mission.
        output = self.output_layer(output)
        # batch_size
        bs = output.shape[0]

        if padding_mask is not None:
            crf_mask = padding_mask.transpose(0, 1).contiguous().byte()  # mask for crf (batch_sequence_max_len, batch_size)
        else:
            crf_mask = None
        # dropout
        output = self.dropout_layer(output)
        # Get the emission scores from Embedding. shape is (seq_length, batch_size, num_tags)
        output = output.transpose(0, 1)

        # 转置label为(seq_length, batch_size)，适配crf的输入
        label_T = label.transpose(0, 1).contiguous()
        # Find the best path, and get the negative_log_likelihood loss according to tags
        loss, predict = self.crf(output, label_T, mask=crf_mask), self.crf.decode(output)
        # 取batch的avg loss
        loss = (-loss) / bs
        # list to tensor must be put into the same device
        predict = torch.LongTensor(predict).to(label.device)

        ### 计算correct
        # 将真实label转化为 1 * x 的矩阵， x表示token数量，也就是个一维的tensor
        label = label.contiguous().view(-1)
        # 同样拉直predict
        predict = predict.contiguous().view(-1)
        label_mask = (label > 0).float().to(label.device)  # label中的元素大于0则转换为1.0，即非padding字符对应的mask为1, padding对应0
        # view(-1) 表示 label_mask拉长为 1 * x 矩阵， x表示token数量
        label_mask = label_mask.contiguous().view(-1)
        predict = predict * label_mask.long()

        correct = torch.sum((predict.eq(label)).float())

        return loss, correct, predict, label

class onlySoftmax(nn.Module):
    def __init__(self, args):
        """

        :param args: 命令行参数的实例对象
        """
        # config
        super(onlySoftmax, self).__init__()
        self.hidden_size = args.hidden_size
        self.dropout = args.dropout
        self.tag_to_ix, self.begin_ids = args.labels_map, args.begin_ids
        self.tagset_size = len(self.tag_to_ix)

        # 创建网络结构
        self.embedding = WordEmbedding(args, len(args.vocab))
        self.dropout_layer = nn.Dropout(p=self.dropout)
        self.output_layer = nn.Linear(self.hidden_size, self.tagset_size)
        self.softmax = nn.LogSoftmax(dim=-1)  # 在最后一维上sum为1

    def forward(self, src, label, padding_mask=None, batch_sequence_max_len=None):
        """
        Args:
            src: means token_ids  [batch_size x seq_length]
            label: means ner label_ids  [batch_size x seq_length]
            padding_mask: [batch_size x seq_length]
            batch_sequence_max_len: int
        Returns:
            loss: Sequence labeling loss.
            correct: Number of labels that are predicted correctly.
            predict: Predicted label.
            label: Gold label.
        """

        # get batch_sequence_max_len
        if batch_sequence_max_len is None or batch_sequence_max_len <= 0:
            batch_sequence_max_len = src.shape[1]
        # reshape输入的数据-基于batch_sequence_max_len动态调整src/label/padding_mask的长度，减少对无用padding的计算
        src = src[:, :batch_sequence_max_len]
        label = label[:, :batch_sequence_max_len]
        if padding_mask is not None:
            padding_mask = padding_mask[:, :batch_sequence_max_len]

        # batch_size
        bs = src.shape[0]

        # Embedding.
        emb = self.embedding(src, None)
        # dropout
        output = self.dropout_layer(emb)
        output = self.output_layer(output)
        # result
        # 通过softmax输出每个token对应各个ner label的概率
        output = self.softmax(output)
        # view(-1, self.tagset_size)指将output转化为 batch token_nums * ner_label_nums 的矩阵
        output = output.contiguous().view(-1, self.tagset_size)

        ###### 拉直data and label，计算loss
        # 将真实label转化为 x * 1的矩阵， x表示token数量，直观上看就是一个label作为新矩阵的一行
        label = label.contiguous().view(-1, 1)
        label_mask = (label > 0).float().to(torch.device(label.device))  # label中的元素大于0则转换为1.0，即非padding字符对应的mask为1
        # one_hot：token_nums * ner_label_nums 的 one hot 矩阵，size与output相同，值为1指示真实标签
        one_hot = torch.zeros(label_mask.size(0), self.tagset_size). \
            to(torch.device(label.device)). \
            scatter_(1, label, 1.0)
        # label smooth
        epsilon = 0.1
        # 平滑后的标签有1-epsilon的概率来自于原分布，有epsilon的概率来自于均匀分布
        label_smooth = (1 - epsilon) * one_hot + epsilon / self.tagset_size

        # * 表示同位置元素相乘
        numerator = -torch.sum(output * label_smooth, 1)
        # label_mask拉长为 1 * x 矩阵， x表示token数量
        label_mask = label_mask.contiguous().view(-1)
        # label拉长为 1 * x 矩阵， x表示token数量
        label = label.contiguous().view(-1)
        # sum  矩阵内所有元素求和，得到分子，表示
        numerator = torch.sum(label_mask * numerator)
        loss = numerator
        predict = output.argmax(dim=-1)
        correct = torch.sum(
            label_mask * (predict.eq(label)).float()
        )

        return loss, correct, predict, label


def getArgs():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Path options. 配置各种路径
    parser.add_argument("--pretrained_model_path", default=None, type=str,
                        help="Path of the pretrained model.")
    parser.add_argument("--output_model_path", default="./models/ner_model.bin", type=str,
                        help="Path of the output model.")
    parser.add_argument("--vocab_path", default="./models/google_vocab.txt", type=str,
                        help="Path of the vocabulary file.")
    parser.add_argument("--config_path", default="./models/google_config.json", type=str,
                        help="Path of the config file.")
    parser.add_argument("--train_path", type=str, required=True,
                        help="Path of the trainset.")
    parser.add_argument("--dev_path", type=str, required=True,
                        help="Path of the devset.")
    parser.add_argument("--test_path", type=str, required=True,
                        help="Path of the testset.")


    # Model options.
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch_size.")
    parser.add_argument("--seq_length", default=512, type=int,
                        help="Sequence length.")
    parser.add_argument("--encoder", choices=["bert", "lstm", "gru", \
                                              "cnn", "gatedcnn", "attn", \
                                              "rcnn", "crnn", "gpt", "bilstm"], \
                        default="bert", help="Encoder type.")
    parser.add_argument("--bidirectional", action="store_true", help="Specific to recurrent model.")

    # Subword options.
    parser.add_argument("--subword_type", choices=["none", "char"], default="none",
                        help="Subword feature type.")
    parser.add_argument("--sub_vocab_path", type=str, default="models/sub_vocab.txt",
                        help="Path of the subword vocabulary file.")
    parser.add_argument("--subencoder", choices=["avg", "lstm", "gru", "cnn"], default="avg",
                        help="Subencoder type.")
    parser.add_argument("--sub_layers_num", type=int, default=2, help="The number of subencoder layers.")

    # Optimizer options.
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                        help="Learning rate.")
    parser.add_argument("--warmup", type=float, default=0.1,
                        help="Warm up value.")

    # Training options.
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Dropout.")
    parser.add_argument("--epochs_num", type=int, default=5,
                        help="Number of epochs.")
    parser.add_argument("--report_steps", type=int, default=100,
                        help="Specific steps to print prompt.")
    parser.add_argument("--seed", type=int, default=7,
                        help="Random seed.")

    # kg
    parser.add_argument("--kg_name", required=True, help="KG name or path")

    args = parser.parse_args()
    return args


def getLabeltoIx(train_path):
    # return a python dict, which is used to convert ner labels to ids

    # 创建供KBERT使用的label to id字典labels_map
    labels_map = {"[PAD]": 0, "[ENT]": 1}
    begin_ids = []

    # Find tagging labels 遍历训练集，找到全部的 ner label 并id化，然后找到每个B label对应的id
    with open(train_path, mode="r", encoding="utf-8") as f:
        for line_id, line in enumerate(f):
            if line_id == 0:
                continue
            labels = line.strip().split("\t")[1].split()
            for l in labels:
                if l not in labels_map:
                    if l.startswith("B") or l.startswith("S"):
                        begin_ids.append(len(labels_map))
                    labels_map[l] = len(labels_map)

    print("original Labels from Dataset: ", labels_map)
    return labels_map, begin_ids


def main():
    """
    main steps in this function:
    1.initialize args
    2.Build knowledge graph
    3.Build sequence labeling model base on pretrain_model, and try to use multiple GPUs, and load sequence labeling model to specified device
    4.define Dataset loader function, read_dataset function and evaluate function
    5.train, dev and eval
    """

    ###############
    # 1.初始化args #
    ###############
    args = getArgs()
    args.labels_map, args.begin_ids = getLabeltoIx(args.train_path)
    args.labels_num = len(args.labels_map)

    # Load the hyperparameters of the config file. 加载bert_config.json
    args = load_hyperparam(args)

    # Load vocabulary. 加载vocab.txt
    vocab = Vocab()
    vocab.load(args.vocab_path)
    args.vocab = vocab
    ##################
    # 1.初始化args完毕 #
    ##################

    # 随机数seed，默认7
    set_seed(args.seed)

    ################
    # 2.Build knowledge graph.
    if args.kg_name == 'none':
        spo_files = []
    else:
        spo_files = [args.kg_name]
    # 加载知识库，返回实例
    kg = KnowledgeGraph(spo_files=spo_files, predicate=False, tokenizer_domain="medicine")


    ######################
    # 3.Build sequence labeling model base on pretrained_model, and try to use multiple GPUs, and load sequence labeling model to specified device
    # Build sequence labeling model. 创建命名实体识别模型的实例对象
    # model = onlyCrf(args)
    # model = lstmCrf(args)
    # model = gruCrf(args)
    # model = onlyLstm(args)
    # model = onlyGru(args)
    model = onlySoftmax(args)

    # labels_map = model.tag_to_ix 不能放在model = nn.DataParallel(model)之后，否则报错'DataParallel' object has no attribute 'tag_to_ix'
    labels_map = model.tag_to_ix

    # 使用多GPU. For simplicity, we use DataParallel wrapper to use multiple GPUs.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 数据并行
    if torch.cuda.device_count() > 1:
        print("{} GPUs are available. Let's use them.".format(torch.cuda.device_count()))
        # 在多个GPU上并行模型，方法是让模型使用 DataParallel 并行运行
        model = nn.DataParallel(model)

    # 模型加载到device(cpu或gpu)上
    model = model.to(device)

    ############################
    # 4.define Dataset loader function, read_dataset function and evaluate function
    # Dataset loader.
    def batch_loader(batch_size, input_ids, label_ids, mask_ids, pos_ids, vm_ids, tag_ids):
        instances_num = input_ids.size()[0]
        for i in range(instances_num // batch_size):
            input_ids_batch = input_ids[i*batch_size: (i+1)*batch_size, :]
            label_ids_batch = label_ids[i*batch_size: (i+1)*batch_size, :]
            mask_ids_batch = mask_ids[i*batch_size: (i+1)*batch_size, :]
            pos_ids_batch = pos_ids[i*batch_size: (i+1)*batch_size, :]
            vm_ids_batch = vm_ids[i*batch_size: (i+1)*batch_size, :, :]
            tag_ids_batch = tag_ids[i*batch_size: (i+1)*batch_size, :]
            yield input_ids_batch, label_ids_batch, mask_ids_batch, pos_ids_batch, vm_ids_batch, tag_ids_batch
        if instances_num > instances_num // batch_size * batch_size:
            input_ids_batch = input_ids[instances_num//batch_size*batch_size:, :]
            label_ids_batch = label_ids[instances_num//batch_size*batch_size:, :]
            mask_ids_batch = mask_ids[instances_num//batch_size*batch_size:, :]
            pos_ids_batch = pos_ids[instances_num//batch_size*batch_size:, :]
            vm_ids_batch = vm_ids[instances_num//batch_size*batch_size:, :, :]
            tag_ids_batch = tag_ids[instances_num//batch_size*batch_size:, :]
            yield input_ids_batch, label_ids_batch, mask_ids_batch, pos_ids_batch, vm_ids_batch, tag_ids_batch

    def batch_loader_bywxx(batch_size, input_ids, label_ids, mask_ids, pos_ids, vm_ids, tag_ids, padding_mask_ids):
        instances_num = input_ids.size()[0]
        for i in range(instances_num // batch_size):
            input_ids_batch = input_ids[i*batch_size: (i+1)*batch_size, :]
            label_ids_batch = label_ids[i*batch_size: (i+1)*batch_size, :]
            mask_ids_batch = mask_ids[i*batch_size: (i+1)*batch_size, :]
            pos_ids_batch = pos_ids[i*batch_size: (i+1)*batch_size, :]
            vm_ids_batch = vm_ids[i*batch_size: (i+1)*batch_size, :, :]
            tag_ids_batch = tag_ids[i*batch_size: (i+1)*batch_size, :]
            padding_mask_ids_batch = padding_mask_ids[i*batch_size: (i+1)*batch_size, :]
            yield input_ids_batch, label_ids_batch, mask_ids_batch, pos_ids_batch, vm_ids_batch, tag_ids_batch, padding_mask_ids_batch
        if instances_num > instances_num // batch_size * batch_size:
            input_ids_batch = input_ids[instances_num//batch_size*batch_size:, :]
            label_ids_batch = label_ids[instances_num//batch_size*batch_size:, :]
            mask_ids_batch = mask_ids[instances_num//batch_size*batch_size:, :]
            pos_ids_batch = pos_ids[instances_num//batch_size*batch_size:, :]
            vm_ids_batch = vm_ids[instances_num//batch_size*batch_size:, :, :]
            tag_ids_batch = tag_ids[instances_num//batch_size*batch_size:, :]
            padding_mask_ids_batch = padding_mask_ids[instances_num//batch_size*batch_size:, :]
            yield input_ids_batch, label_ids_batch, mask_ids_batch, pos_ids_batch, vm_ids_batch, tag_ids_batch, padding_mask_ids_batch

    # Read dataset and convert. convert tokens to token_ids and labels to label_ids, and generate mask by [1] * len(token_ids)
    def read_dataset(path):
        dataset = []
        with open(path, mode="r", encoding="utf-8") as f:
            # 首行是header：text\tlabel，所以先读一行来去掉header
            f.readline()
            # tokens such as "呈 三 组 （ 5 / 1 3 个 ） 淋 巴 结 癌 转 移 。"
            # labels such as "O B-Anatomy I-Anatomy I-Anatomy I-Anatomy I-Anatomy I-Anatomy I-Anatomy I-Anatomy I-Anatomy I-Anatomy I-Anatomy I-Anatomy O O O O"
            tokens, labels = [], []
            for line_id, line in enumerate(f):
                tokens, labels = line.strip().split("\t")

                text = ''.join(tokens.split(" "))
                # 融入知识库，返回融合后的tokens, soft-pos, vm, tag
                tokens, pos, vm, tag = kg.add_knowledge_with_vm([text], add_pad=True, max_length=args.seq_length)
                tokens = tokens[0]
                pos = pos[0]
                vm = vm[0].astype("bool")
                tag = tag[0]

                # tokens to ids
                tokens = [vocab.get(t) for t in tokens]
                # ner labels to ids
                labels = [labels_map[l] for l in labels.split(" ")]
                mask = [1] * len(tokens)

                new_labels = []
                j = 0
                for i in range(len(tokens)):
                    if tag[i] == 0 and tokens[i] != PAD_ID:
                        new_labels.append(labels[j])
                        j += 1
                    elif tag[i] == 1 and tokens[i] != PAD_ID:  # tag[i] == 1表示是从知识库添加的实体，tokens[i] != PAD_ID表示不是padding符
                        new_labels.append(labels_map['[ENT]'])
                    else:
                        new_labels.append(labels_map[PAD_TOKEN])  # labels_map[PAD_TOKEN]为0，代表PAD填充

                # 每个样本用[tokens, new_labels, mask, pos, vm, tag]表示，其中tag用来区分原句子和引入的知识库实体，知识库实体用1表示，其他用0
                dataset.append([tokens, new_labels, mask, pos, vm, tag])
        # 打乱数据集
        random.shuffle(dataset)
        return dataset

    # 调用kg.add_knowledge_with_vm_bywxx，返回[[tokens, new_labels, seg_mask, pos, vm, tag, padding_mask], ...]
    def read_dataset_bywxx(path):
        dataset = []
        with open(path, mode="r", encoding="utf-8") as f:
            # 首行是header：text\tlabel，所以先读一行来去掉header
            f.readline()
            # tokens such as "呈 三 组 （ 5 / 1 3 个 ） 淋 巴 结 癌 转 移 。"
            # labels such as "O B-Anatomy I-Anatomy I-Anatomy I-Anatomy I-Anatomy I-Anatomy I-Anatomy I-Anatomy I-Anatomy I-Anatomy I-Anatomy I-Anatomy O O O O"
            tokens, labels = [], []
            total_kg_entity_cnt = 0
            for line_id, line in enumerate(f):
                tokens, labels = line.strip().split("\t")

                text = ''.join(tokens.split(" "))
                # 融入知识库，返回融合后的tokens, soft-pos, vm, tag
                tokens, pos, vm, tag, padding_mask, entity_num = kg.add_knowledge_with_vm_bywxx([text], max_length=args.seq_length)
                tokens = tokens[0]
                pos = pos[0]
                vm = vm[0].astype("bool")
                tag = tag[0]
                padding_mask = padding_mask[0]
                total_kg_entity_cnt += entity_num

                # tokens to ids
                tokens = [vocab.get(t) for t in tokens]
                # ner labels to ids
                labels = [labels_map[l] for l in labels.split(" ")]
                seg_mask = [1] * len(tokens)

                new_labels = []
                j = 0
                for i in range(len(tokens)):
                    if tag[i] == 0 and tokens[i] != PAD_ID:
                        new_labels.append(labels[j])
                        j += 1
                    elif tag[i] == 1 and tokens[i] != PAD_ID:  # tag[i] == 1表示是从知识库添加的实体，tokens[i] != PAD_ID表示不是padding符
                        new_labels.append(labels_map['[ENT]'])
                    else:
                        new_labels.append(labels_map[PAD_TOKEN])  # labels_map[PAD_TOKEN]为0，代表PAD填充

                # 每个样本用[tokens, new_labels, seg_mask, pos, vm, tag, padding_mask]表示，其中tag用来区分原句子和引入的知识库实体，知识库实体用1表示，其他用0
                dataset.append([tokens, new_labels, seg_mask, pos, vm, tag, padding_mask])
        print('add %s kg_entity to dataset' % str(total_kg_entity_cnt))
        # 打乱数据集
        random.shuffle(dataset)
        return dataset

    # Evaluation function. return f1-value
    def evaluate(args, is_test):
        if is_test:
            dataset = read_dataset_bywxx(args.test_path)
        else:
            dataset = read_dataset_bywxx(args.dev_path)

        input_ids = torch.LongTensor([sample[0] for sample in dataset])
        label_ids = torch.LongTensor([sample[1] for sample in dataset])
        mask_ids = torch.LongTensor([sample[2] for sample in dataset])
        pos_ids = torch.LongTensor([sample[3] for sample in dataset])
        vm_ids = torch.BoolTensor([sample[4] for sample in dataset])
        tag_ids = torch.LongTensor([sample[5] for sample in dataset])
        padding_mask_ids = torch.LongTensor([sample[6] for sample in dataset])

        instances_num = input_ids.size(0)
        batch_size = args.batch_size

        if is_test:
            print('testing')
            print("Batch size: ", batch_size)
            print("The number of test instances:", instances_num)
        else:
            print('dev-ing')
            print("Batch size: ", batch_size)
            print("The number of dev instances:", instances_num)
 
        correct = 0  # 实体类别及边界正确
        tag_correct = 0   # 单个token的tag预测正确
        gold_entities_num = 0
        pred_entities_num = 0

        # 创建混淆矩阵
        confusion = torch.zeros(len(labels_map), len(labels_map), dtype=torch.long)

        # 不启用 BatchNormalization 和 Dropout，保证BN和dropout不发生变化
        # pytorch框架会自动把BN和Dropout固定住，不会取平均，而是用训练好的值，不然的话，一旦dev or test的batch_size过小，很容易就会被BN层影响结果。
        model.eval()

        # 对每个batch中每个sentence，对比实际的实体边界和预测的实体边界，单个实体边界完全一致则correct++
        for i, (input_ids_batch, label_ids_batch, mask_ids_batch, pos_ids_batch, vm_ids_batch, tag_ids_batch, padding_mask_ids_batch) in enumerate(batch_loader_bywxx(batch_size, input_ids, label_ids, mask_ids, pos_ids, vm_ids, tag_ids, padding_mask_ids)):

            input_ids_batch = input_ids_batch.to(device)
            label_ids_batch = label_ids_batch.to(device)
            padding_mask_ids_batch = padding_mask_ids_batch.to(device)
            batch_true_max_len = int(torch.max(torch.sum(padding_mask_ids_batch, dim=1)).item())

            # forward!!!
            loss, _, pred, gold = model(input_ids_batch, label_ids_batch, padding_mask_ids_batch, batch_true_max_len)
            # pred预测值， gold真实值

            # 填充混淆矩阵
            for jj in range(pred.size()[0]):
                confusion[pred[jj], gold[jj]] += 1
            tag_correct += torch.sum(pred == gold).item()

            for j in range(gold.size()[0]):
                if gold[j].item() in args.begin_ids:
                    gold_entities_num += 1
 
            for j in range(pred.size()[0]):
                # 在非[PAD]字符下预测为实体起点
                if pred[j].item() in args.begin_ids and gold[j].item() != args.labels_map["[PAD]"]:
                    pred_entities_num += 1

            pred_entities_pos = []
            gold_entities_pos = []
            start, end = 0, 0

            # 查找真实实体的起点和终点
            for j in range(gold.size()[0]):
                if gold[j].item() in args.begin_ids:
                    start = j
                    for k in range(j+1, gold.size()[0]):
                        
                        if gold[k].item() == args.labels_map['[ENT]']:
                            continue

                        if gold[k].item() == args.labels_map["[PAD]"] or gold[k].item() == args.labels_map["O"] or gold[k].item() in args.begin_ids:
                            end = k - 1
                            break
                    else:
                        end = gold.size()[0] - 1
                    # gold_entities_pos.append((start, end))
                    gold_entities_pos.append((start, end, gold[start].item()))   # wxx (实体起点，实体终点，实体类型)

            # 查找预测实体的起点和终点
            for j in range(pred.size()[0]):
                if pred[j].item() in args.begin_ids and gold[j].item() != args.labels_map["[PAD]"] and gold[j].item() != args.labels_map["[ENT]"]:
                    start = j
                    for k in range(j+1, pred.size()[0]):

                        if gold[k].item() == args.labels_map['[ENT]']:
                            continue

                        if pred[k].item() == args.labels_map["[PAD]"] or pred[k].item() == args.labels_map["O"] or pred[k].item() in args.begin_ids:
                            end = k - 1
                            break
                    else:
                        end = pred.size()[0] - 1
                    # pred_entities_pos.append((start, end))
                    pred_entities_pos.append((start, end, pred[start].item()))  # wxx (实体起点，实体终点，实体类型)

            # 预测实体的起点终点位置相同，视为正确
            for entity in pred_entities_pos:
                if entity not in gold_entities_pos:
                    continue
                else: 
                    correct += 1

        print("Confusion matrix:")
        print(confusion)
        print("Report precision, recall, and f1:")
        for ii in range(confusion.size()[0]):
            # 完善代码，避免ZeroDivisionError: division by zero. wxx
            p_denominator = confusion[ii, :].sum().item()
            if p_denominator != 0:
                p = confusion[ii, ii].item() / p_denominator
            else:
                p = 0
            r_denominator = confusion[:, ii].sum().item()
            if r_denominator != 0:
                r = confusion[ii, ii].item() / r_denominator
            else:
                r = 0
            if p + r != 0:
                f1 = 2 * p * r / (p + r)
            else:
                f1 = 0
            # 注意，取label_1_f1 = f1只适用于2分类问题
            if ii == 1:
                label_1_f1 = f1
            print("Label {}: {:.5f}, {:.5f}, {:.5f}".format(ii, p, r, f1))

        print("correct, pred_entities_num, gold_entities_num are {:.5f}, {:.5f}, {:.5f}".format(correct, pred_entities_num, gold_entities_num))
        print("Report precision, recall, and f1:")
        if pred_entities_num == 0:
            p = 0
        else:
            p = correct/pred_entities_num
        if gold_entities_num == 0:
            r = 0
        else:
            r = correct/gold_entities_num
        if p + r == 0:
            f1 = 0
        else:
            f1 = 2*p*r/(p+r)
        print("{:.5f}, {:.5f}, {:.5f}".format(p, r, f1))

        return f1

    ########################
    # 5.train, dev and eval
    # Training phase.
    print("Start training.")
    print('loading train_data')
    # instances = read_dataset(args.train_path)
    instances = read_dataset_bywxx(args.train_path)

    # torch.LongTensor means 64-bit integer (signed)
    input_ids = torch.LongTensor([ins[0] for ins in instances])
    label_ids = torch.LongTensor([ins[1] for ins in instances])
    mask_ids = torch.LongTensor([ins[2] for ins in instances])
    pos_ids = torch.LongTensor([ins[3] for ins in instances])
    vm_ids = torch.BoolTensor([ins[4] for ins in instances])
    tag_ids = torch.LongTensor([ins[5] for ins in instances])
    padding_mask_ids = torch.LongTensor([ins[6] for ins in instances])

    instances_num = input_ids.size(0)
    batch_size = args.batch_size
    train_steps = int(instances_num * args.epochs_num / batch_size) + 1

    print("Batch size: ", batch_size)
    print("The number of training instances:", instances_num)
    print('loading train_data finished')
    print('-------------------------------------')

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    # A.权重衰减（weight decay）,即l2正则化，让权重衰减到更小的值，在一定程度上减少模型过拟合的问题
    optimizer_grouped_parameters = [
                {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.01},
                {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.0}
    ]

    ### optimizer for bertless model ###
    optimizer = torch.optim.Adam(optimizer_grouped_parameters, lr=args.learning_rate)
    ### optimizer for bertless model ###

    # 设置scheduler实现动态学习率
    # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=[lambda1, lambda2])  #
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)

    total_loss = 0.0
    f1 = 0.0
    best_f1 = 0.0

    for epoch in range(1, args.epochs_num+1):
        epoch_loss = 0.0
        # Sets the module in training mode.
        model.train()
        # for i, (input_ids_batch, label_ids_batch, mask_ids_batch, pos_ids_batch, vm_ids_batch, tag_ids_batch) in enumerate(batch_loader(batch_size, input_ids, label_ids, mask_ids, pos_ids, vm_ids, tag_ids)):
        for i, (input_ids_batch, label_ids_batch, mask_ids_batch, pos_ids_batch, vm_ids_batch, tag_ids_batch, padding_mask_ids_batch) in enumerate(batch_loader_bywxx(batch_size, input_ids, label_ids, mask_ids, pos_ids, vm_ids, tag_ids, padding_mask_ids)):
            model.zero_grad()

            input_ids_batch = input_ids_batch.to(device)
            label_ids_batch = label_ids_batch.to(device)
            mask_ids_batch = mask_ids_batch.to(device)
            pos_ids_batch = pos_ids_batch.to(device)
            tag_ids_batch = tag_ids_batch.to(device)
            vm_ids_batch = vm_ids_batch.long().to(device)
            padding_mask_ids_batch = padding_mask_ids_batch.to(device)
            batch_true_max_len = int(torch.max(torch.sum(padding_mask_ids_batch, dim=1)).item())

            loss, _, _, _ = model(input_ids_batch, label_ids_batch, padding_mask_ids_batch, batch_true_max_len)
            if hasattr(model, "module") and torch.cuda.device_count() > 1:   # model被DataParallel加载会使得model增加attr "module"
                loss = torch.mean(loss)   # mean() to average on multi-gpu.
            total_loss += loss.item()
            epoch_loss += loss.item()
            if (i + 1) % args.report_steps == 0:  # iterate args.report_steps次，输出一次total_loss
                print("Epoch id: {}, Training steps: {}, Avg loss: {:.3f}".format(epoch, i+1, total_loss / args.report_steps))
                total_loss = 0.
            loss.backward()
            # 加入梯度剪裁，防止梯度爆炸
            nn.utils.clip_grad_value_(model.parameters(), clip_value=2.0)   # 按值剪裁
            optimizer.step()

        print("Epoch {} total loss is {:.3f}".format(epoch, epoch_loss))
        # scheduler.step(epoch_loss)

        # # Evaluation phase.
        # print("Start evaluate on dev dataset.")
        # f1 = evaluate(args, False)
        print("Start evaluation on test dataset.")
        f1 = evaluate(args, True)

        if f1 > best_f1:
            best_f1 = f1
            save_model(model, args.output_model_path)
        else:
            continue

    # Evaluation phase.
    print("Final evaluation on test dataset.")

    if hasattr(model, "module") and torch.cuda.device_count() > 1:   # wxx
        model.module.load_state_dict(torch.load(args.output_model_path))
    else:
        model.load_state_dict(torch.load(args.output_model_path))

    evaluate(args, True)


if __name__ == "__main__":
    main()
