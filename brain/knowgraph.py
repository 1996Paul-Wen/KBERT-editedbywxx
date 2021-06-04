# coding: utf-8
"""
KnowledgeGraph
"""
import os
import brain.config as config
import pkuseg
import numpy as np


class KnowledgeGraph(object):
    """
    spo_files - list of Path of *.spo files, or default kg name. e.g., ['HowNet']
    接收一个list of Path of *.spo files，说明可以融入一个或多个知识库
    """

    def __init__(self, spo_files, predicate=False, tokenizer_domain="default"):
        self.predicate = predicate
        self.tokenizer_domain = tokenizer_domain
        self.spo_file_paths = [config.KGS.get(f, f) for f in spo_files]
        self.lookup_table = self._create_lookup_table()
        # 基于lookup_table.keys() + config.NEVER_SPLIT_TAG 使用知识库中的三元组主语，构造词典作为tokenize时的用户字典
        self.segment_vocab = list(self.lookup_table.keys()) + config.NEVER_SPLIT_TAG
        # 调用pkuseg实例化tokenizer器，model_name支持"default" "news" "web" "medicine" "tourism" 多个领域
        self.tokenizer = pkuseg.pkuseg(model_name=self.tokenizer_domain, postag=False, user_dict=self.segment_vocab)
        self.special_tags = set(config.NEVER_SPLIT_TAG)

    def _create_lookup_table(self):
        # 根据self.spo_file_paths，创建一个字典lookup_table，形如{subject_str: object_set}, such as {'姚明': {'火箭队','上海大鲨鱼'}}
        lookup_table = {}
        for spo_path in self.spo_file_paths:
            print("[KnowledgeGraph] Loading spo from {}".format(spo_path))
            with open(spo_path, 'r', encoding='utf-8') as f:
                cnt = 0
                for line in f:
                    try:
                        # 分割三元组，形成主 谓 宾
                        subj, pred, obje = line.strip().split("\t")    
                    except:
                        print("[KnowledgeGraph] Bad spo:", line, cnt)
                        
                    # if self.predicate is true, value保留谓语，否则舍弃
                    if self.predicate:
                        value = pred + obje
                    else:
                        value = obje
                    if subj in lookup_table.keys():
                        lookup_table[subj].add(value)
                    else:
                        lookup_table[subj] = set([value])
                    cnt += 1
        return lookup_table

    def add_knowledge_with_vm(self, sent_batch, max_entities=config.MAX_ENTITIES, max_length=128):
        """
        input: sent_batch - list of sentences, e.g., ["abcd", "efgh"] or ["姚明在休斯顿火箭队", "科比在湖人队"]
        return: know_sent_batch - list of sentences with entites embedding
                position_batch - list of position index of each character. 字符级别的软位置编码列表
                visible_matrix_batch - list of visible matrixs
                seg_batch - list of segment tags
        """
        # 将batch中的每个sentence进行分词
        split_sent_batch = [self.tokenizer.cut(sent) for sent in sent_batch]
        know_sent_batch = []
        position_batch = []
        visible_matrix_batch = []
        seg_batch = []
        for split_sent in split_sent_batch:
            
            # split_sent such as ['患者', '3月', '前', '因', '"', '直肠癌', '"', '于', '我院', '于', '全麻', '上行', '直肠', '癌根', '治术']
            
            # create tree
            sent_tree = []
            pos_idx_tree = []
            abs_idx_tree = []
            ### 维护两个变量完成软硬位置编码
            # pos_idx表示在不引入外部知识库下，原始token的位置编码，即soft-position index
            pos_idx = -1
            # abs_idx绝对编码，对应KBert论文中的hard-position index
            abs_idx = -1
            abs_idx_src = []
            for token in split_sent:

                entities = list(self.lookup_table.get(token, []))[:max_entities]
                sent_tree.append((token, entities))
                
                # 1.处理token的软硬编码
                if token in self.special_tags:
                    token_pos_idx = [pos_idx+1]
                    token_abs_idx = [abs_idx+1]
                else:
                    token_pos_idx = [pos_idx+i for i in range(1, len(token)+1)]
                    token_abs_idx = [abs_idx+i for i in range(1, len(token)+1)]
                abs_idx = token_abs_idx[-1]
                
                # 2.处理基于token从外部知识库引入的entities的软硬编码
                entities_pos_idx = []
                entities_abs_idx = []
                for ent in entities:
                    # 利用token_pos_idx[-1]，紧接着当前token的末尾position index完成【软位置编码】
                    ent_pos_idx = [token_pos_idx[-1] + i for i in range(1, len(ent)+1)]
                    entities_pos_idx.append(ent_pos_idx)
                    # 利用abs_idx完成【硬位置编码】
                    ent_abs_idx = [abs_idx + i for i in range(1, len(ent)+1)]
                    abs_idx = ent_abs_idx[-1]
                    entities_abs_idx.append(ent_abs_idx)

                # soft-position index
                pos_idx_tree.append((token_pos_idx, entities_pos_idx))
                pos_idx = token_pos_idx[-1]
                # hard-position index
                abs_idx_tree.append((token_abs_idx, entities_abs_idx))
                abs_idx_src += token_abs_idx

            # Get know_sent and pos
            know_sent = []
            pos = []
            seg = []
            for i in range(len(sent_tree)):
                # sent_tree[i][0]就是原sentence中的token
                word = sent_tree[i][0]
                if word in self.special_tags:
                    know_sent += [word]
                    seg += [0]
                else:
                    add_word = list(word)
                    know_sent += add_word 
                    seg += [0] * len(add_word)
                pos += pos_idx_tree[i][0]
                
                for j in range(len(sent_tree[i][1])):
                    # token在知识库中找到的知识实体词，赋给add_word
                    add_word = list(sent_tree[i][1][j])
                    know_sent += add_word
                    seg += [1] * len(add_word)
                    pos += list(pos_idx_tree[i][1][j])

            token_num = len(know_sent)

            # Calculate visible matrix
            # 可见矩阵的加入使得对当前词进行编码时候，模型能‘看得见’当前词枝干上的信息,而‘看不见’与当前词不相干的其他枝干上的信息,以防不相干枝干在self-attention计算时候互相影响
            visible_matrix = np.zeros((token_num, token_num))
            for item in abs_idx_tree:
                src_ids = item[0]
                for id in src_ids:
                    visible_abs_idx = abs_idx_src + [idx for ent in item[1] for idx in ent]
                    visible_matrix[id, visible_abs_idx] = 1
                for ent in item[1]:
                    for id in ent:
                        visible_abs_idx = ent + src_ids
                        visible_matrix[id, visible_abs_idx] = 1

            # 截取max_length长度
            src_length = len(know_sent)
            if len(know_sent) < max_length:
                pad_num = max_length - src_length
                know_sent += [config.PAD_TOKEN] * pad_num
                seg += [0] * pad_num
                pos += [max_length - 1] * pad_num
                visible_matrix = np.pad(visible_matrix, ((0, pad_num), (0, pad_num)), 'constant')  # pad 0
            else:
                know_sent = know_sent[:max_length]
                seg = seg[:max_length]
                pos = pos[:max_length]
                visible_matrix = visible_matrix[:max_length, :max_length]
            
            know_sent_batch.append(know_sent)  # know_sent such as ['[CLS]', '患', '者', '有', '青', '光', '眼', '高','眼','压', '，', '看', '不', '清', '。', '[SEP]'], 其中【高眼压】为知识库补充
            position_batch.append(pos)  # pos such as [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 7, 8, 9, 10, 11, 12]
            visible_matrix_batch.append(visible_matrix)   # token_num * token_num matrix
            seg_batch.append(seg)  # seg such as [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0]
        
        return know_sent_batch, position_batch, visible_matrix_batch, seg_batch

    # add_knowledge_with_vm_bywxx与add_knowledge_with_vm的差异
    # 1.融入句子的外部知识实体以@标记开头，以#标记结尾
    # 2.返回值增加padding_mask_batch，用以区别sentence token和不足max_length的padding部分
    # 3.返回值增加entity_num，表示sent_batch融入的实体数量
    def add_knowledge_with_vm_bywxx(self, sent_batch, max_entities=config.MAX_ENTITIES, max_length=128):
        """
        # 在原add_knowledge_with_vm方法的基础上，给sentence添加知识实体构成sentence tree时，加入的实体以@开头，以#结尾，便于模型识别这是外部实体
        input: sent_batch - list of sentences, e.g., ["abcd", "efgh"] or ["姚明在休斯顿火箭队", "科比在湖人队"]
        return: know_sent_batch - list of sentences with entites embedding
                position_batch - list of position index of each character. 字符级别的软位置编码列表
                visible_matrix_batch - list of visible matrixs
                seg_batch - list of segment tags
                padding_mask_batch - list of padding tags
                entity_num - int
        """
        # 将batch中的每个sentence进行分词
        split_sent_batch = [self.tokenizer.cut(sent) for sent in sent_batch]
        know_sent_batch = []
        position_batch = []
        visible_matrix_batch = []
        seg_batch = []
        padding_mask_batch = []
        entity_num = 0  # sent_batch融入的实体数量
        for split_sent in split_sent_batch:

            # split_sent such as ['患者', '3月', '前', '因', '"', '直肠癌', '"', '于', '我院', '于', '全麻', '上行', '直肠', '癌根', '治术']

            # create tree
            sent_tree = []
            pos_idx_tree = []
            abs_idx_tree = []
            ### 维护两个变量完成软硬位置编码
            # pos_idx表示在不引入外部知识库下，原始token的位置编码，即soft-position index
            pos_idx = -1
            # abs_idx绝对编码，对应KBert论文中的hard-position index
            abs_idx = -1
            abs_idx_src = []
            for token in split_sent:

                entities = list(self.lookup_table.get(token, []))[:max_entities]
                entity_num += len(entities)
                # 给entity加上@前缀和#后缀，使其有明显的分隔符 wxx
                for i in range(len(entities)):
                    ent = entities[i]
                    entities[i] = '@'+ent+'#'
                sent_tree.append((token, entities))

                # 1.处理token的软硬编码
                if token in self.special_tags:
                    token_pos_idx = [pos_idx + 1]
                    token_abs_idx = [abs_idx + 1]
                else:
                    token_pos_idx = [pos_idx + i for i in range(1, len(token) + 1)]
                    token_abs_idx = [abs_idx + i for i in range(1, len(token) + 1)]
                abs_idx = token_abs_idx[-1]

                # 2.处理基于token从外部知识库引入的entities的软硬编码
                entities_pos_idx = []
                entities_abs_idx = []
                for ent in entities:
                    # 利用token_pos_idx[-1]，紧接着当前token的末尾position index完成【软位置编码】
                    ent_pos_idx = [token_pos_idx[-1] + i for i in range(1, len(ent) + 1)]
                    entities_pos_idx.append(ent_pos_idx)
                    # 利用abs_idx完成【硬位置编码】
                    ent_abs_idx = [abs_idx + i for i in range(1, len(ent) + 1)]
                    abs_idx = ent_abs_idx[-1]
                    entities_abs_idx.append(ent_abs_idx)

                # soft-position index
                pos_idx_tree.append((token_pos_idx, entities_pos_idx))
                pos_idx = token_pos_idx[-1]
                # hard-position index
                abs_idx_tree.append((token_abs_idx, entities_abs_idx))
                abs_idx_src += token_abs_idx

            # Get know_sent and pos
            know_sent = []
            pos = []
            # seg是用来区分know_sent中原始token和外部知识实体token的。原始token用0标记，外部token用1标记
            seg = []
            for i in range(len(sent_tree)):
                # sent_tree[i][0]就是原sentence中的token
                word = sent_tree[i][0]
                if word in self.special_tags:
                    know_sent += [word]
                    seg += [0]
                else:
                    add_word = list(word)
                    know_sent += add_word
                    seg += [0] * len(add_word)
                pos += pos_idx_tree[i][0]

                for j in range(len(sent_tree[i][1])):
                    # token在知识库中找到的知识实体词，赋给add_word
                    add_word = list(sent_tree[i][1][j])
                    know_sent += add_word
                    seg += [1] * len(add_word)
                    pos += list(pos_idx_tree[i][1][j])

            # 融入外部知识实体后的句子长度
            token_num = len(know_sent)
            # padding_mask，用于区别token和padding，1表示token，0表示padding，such as [1,1,1,1,0,0]
            padding_mask = [1] * token_num

            # Calculate visible matrix
            # 可见矩阵的加入使得对当前词进行编码时候，模型能‘看得见’当前词枝干上的信息,而‘看不见’与当前词不相干的其他枝干上的信息,以防不相干枝干在self-attention计算时候互相影响
            visible_matrix = np.zeros((token_num, token_num))
            for item in abs_idx_tree:
                src_ids = item[0]
                for id in src_ids:
                    visible_abs_idx = abs_idx_src + [idx for ent in item[1] for idx in ent]
                    visible_matrix[id, visible_abs_idx] = 1
                for ent in item[1]:
                    for id in ent:
                        visible_abs_idx = ent + src_ids
                        visible_matrix[id, visible_abs_idx] = 1

            # 截取max_length长度
            src_length = len(know_sent)
            if len(know_sent) < max_length:
                pad_num = max_length - src_length
                know_sent += [config.PAD_TOKEN] * pad_num
                padding_mask += [0] * pad_num
                seg += [0] * pad_num
                pos += [max_length - 1] * pad_num
                visible_matrix = np.pad(visible_matrix, ((0, pad_num), (0, pad_num)), 'constant')  # pad 0
            else:
                know_sent = know_sent[:max_length]
                padding_mask = padding_mask[:max_length]
                seg = seg[:max_length]
                pos = pos[:max_length]
                visible_matrix = visible_matrix[:max_length, :max_length]

            know_sent_batch.append(
                know_sent)  # know_sent such as ['患', '者', '有', '青', '光', '眼', '高','眼','压', '，', '看', '不', '清', '。'], 其中【高眼压】为知识库补充
            position_batch.append(pos)  # pos such as [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 7, 8, 9, 10, 11, 12]
            visible_matrix_batch.append(visible_matrix)  # token_num * token_num matrix
            seg_batch.append(seg)  # seg such as [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0]
            padding_mask_batch.append(padding_mask)  # sentence_mask such as [1,1,1,1,1,1,0,0,0]

        return know_sent_batch, position_batch, visible_matrix_batch, seg_batch, padding_mask_batch, entity_num
