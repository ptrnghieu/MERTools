from torch.utils.data import Dataset

from .feat_data import Data_Feat
from .feat_data_topn import Data_Feat_TOPN

# 目标：输入 (names, labels, data_type)，得到所有特征与标签
class get_datasets(Dataset):

    def __init__(self, args, names, labels):

        MODEL_DATASET_MAP = {
            
            # 解析特征
            'attention': Data_Feat,
            'lf_dnn': Data_Feat,
            'lmf': Data_Feat,
            'misa': Data_Feat,
            'mmim': Data_Feat,
            'tfn': Data_Feat,
            'mfn': Data_Feat,
            'graph_mfn': Data_Feat,
            'ef_lstm': Data_Feat, 
            'mfm': Data_Feat,
            'mctn': Data_Feat,
            'mult': Data_Feat,


            # 兼容多特征输入
            'attention_topn': Data_Feat_TOPN,

            # cross-role transferability
            'cross_role_attention': Data_Feat,
            'two_stage_model': Data_Feat,
            'grasp_sequence_fusion': Data_Feat,
            'speaker_listener_fusion': Data_Feat,
            'memocmt_fusion': Data_Feat,
            'memocmt_v1': Data_Feat,
            'memocmt_mmin': Data_Feat,
            'memocmt_mmin_hybrid': Data_Feat,
            'memocmt_v2': Data_Feat,
            'memocmt_v3': Data_Feat,
            'memocmt_v4': Data_Feat,
            'memocmt_v5': Data_Feat,
            'memocmt_v6': Data_Feat,
            'memocmt_v7': Data_Feat,
            'memocmt_v8': Data_Feat,
            'memocmt_v9': Data_Feat,
            'memocmt_v10': Data_Feat,
            'memocmt_v11': Data_Feat,
            'memocmt_v12': Data_Feat,
            'memocmt_v13': Data_Feat,
            'memocmt_v14': Data_Feat,
            'memocmt_v15': Data_Feat,
        }

        self.dataset_class = MODEL_DATASET_MAP[args.model]
        self.dataset = self.dataset_class(args, names, labels)

    def __len__(self):
        return self.dataset.__len__()

    def __getitem__(self, index):
        return self.dataset.__getitem__(index)

    def collater(self, instances):
        return self.dataset.collater(instances)
         
    def get_featdim(self):
        return self.dataset.get_featdim()