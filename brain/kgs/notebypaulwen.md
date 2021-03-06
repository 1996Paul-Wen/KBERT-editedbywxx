- **HowNet.spo、Medical.spo**为K-BERT开源项目自带的三元组形式知识图谱
- **Medical-clean.spo**为自建的三元组形式知识图谱，Medical.spo经过以下2步处理后，得到Medical-clean.spo
  - 1.原Medical.spo中，有311个实体存在2组及以上的三元组关系，如' 脑干梗塞': ['疾病', '症状'],'月经量少': ['症状', '疾病']等。现根据CCKS2019医学命名实体识别的任务需要进行了修正，保证一个实体只存在一个三元组关系。311个重复实体和部分修正样例见medicalrepeatentities.txt
  - 2.原Medical.spo中，部分实体的前后或内部参杂空格，如' 脑干梗塞'，'冷球蛋 白血症'等，对这些空格进行去除。
- **Medical-plus.spo**为自建的三元组形式知识图谱，由Medical-clean.spo经过以下步骤处理得到
  - 额外整合GitHub开源项目QASystemOnMedicalGraph提供的疾病实体词2866个和症状实体词1525个，和QASystemOnMedicalKG提供的药物实体3792个、疾病实体49个、症状实体655个
  - Medical-clean.spo中，药物和手术都连接到【治疗】。根据命名实体识别的任务需要进行细化至药物、手术和一般治疗，例如：营养脑神经药物	类别	治疗-头孢替唑	类别	治疗【药物】，右股骨颈骨折髋关节股骨头表面置换术	类别	治疗-左侧人工股骨头置换术	类别	手术【手术】
  - CCKS2019中区别了检查实体和检验实体，而Medical-clean.spo中则未作区分。考虑到区分检查和检验需要更加完备的医学知识，本人知识和能力有限，难以胜任，因此直接删除了Medical-clean.spo中的检查三元组。而保留Medical-clean.spo中的检查三元组的知识图谱，则对应**Medical-plus_with_check_entity.spo**