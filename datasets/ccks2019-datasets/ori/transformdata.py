# -*- coding: utf-8 -*-
# 此脚本进行BIO标注

trans_dict = {
    '疾病和诊断': 'Dis&Diag',
    '药物': 'Drug',
    '实验室检验': 'LabCHeck',
    '解剖部位': 'Anatomy',
    '影像检查': 'MedImgCheck',
    '手术': 'Surgery',
}


with open('train_ori.txt', 'r', encoding='utf-8') as f:
    datas = []
    while True:
        line = f.readline()
        if not line:
            break
        
        content = eval(line.strip())
        # 病例文本
        text = content['originalText'].replace(' ', '*')
        text_list = [char for char in text]
        # 实体位置信息
        entities = content['entities']
        bio = ['O' for i in range(len(text))]
        for en in entities:
            start_pos = en['start_pos']
            end_pos = en['end_pos']
            type = trans_dict[en['label_type']]
            for i in range(start_pos, end_pos):
                if i == start_pos:
                    bio[i] = 'B-' + type
                else:
                    bio[i] = 'I-' + type
        # 打印至屏幕的辅助信息ans # 
        ans = ''
        for i in range(len(bio)):
            ans += (text_list[i] + bio[i] + ' ')
        print(ans)
        
        post_line = ' '.join(text_list) + '	' + ' '.join(bio) + '\n'
        datas.append(post_line)



with open('train.tsv', 'w', encoding='utf-8') as f1:
    f1.write('text_a	label\n')
    f1.writelines(datas[:800])

with open('dev.tsv', 'w', encoding='utf-8') as f0:
    f0.write('text_a	label\n')
    f0.writelines(datas[800:])

with open('test_ori.txt', 'r', encoding='utf-8') as ff:
    with open('test.tsv', 'w', encoding='utf-8') as f2:
        while True:
            line = ff.readline()
            if not line:
                break
            
            content = eval(line.strip())
            # 病例文本
            text = content['originalText'].replace(' ', '*')
            text_list = [char for char in text]
            # 实体位置信息
            entities = content['entities']
            bio = ['O' for i in range(len(text))]
            for en in entities:
                start_pos = en['start_pos']
                end_pos = en['end_pos']
                type = trans_dict[en['label_type']]
                for i in range(start_pos, end_pos):
                    if i == start_pos:
                        bio[i] = 'B-' + type
                    else:
                        bio[i] = 'I-' + type
            ans = ''
            for i in range(len(bio)):
                ans += (text_list[i] + bio[i] + ' ')
            print(ans)
            post_line = ' '.join(text_list) + '	' + ' '.join(bio) + '\n'
            f2.write(post_line)