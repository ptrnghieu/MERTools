import math
import torch
import numpy as np
from torch.utils.data import Dataset
from toolkit.utils.read_data import *

class Data_Feat(Dataset):
    def __init__(self, args, names, labels):

        # analyze path
        self.names = names
        self.labels = labels
        feat_root  = config.PATH_TO_FEATURES[args.dataset]
        if args.snr is None: # 通过snr控制特征读取位置
            audio_root = os.path.join(feat_root, args.audio_feature)
            text_root  = os.path.join(feat_root, args.text_feature )
            video_root = os.path.join(feat_root, args.video_feature)
        else:
            # data2vec-audio-base-960h-UTT -> data2vec-audio-base-960h-noisesnrmix-UTT
            # eGeMAPS_UTT -> eGeMAPS_noisesnrmix_UTT
            audio_root = os.path.join(feat_root, args.audio_feature[-4].join([args.audio_feature[:-4], args.snr, 'UTT']))
            text_root  = os.path.join(feat_root, args.text_feature[-4].join([args.text_feature[:-4], args.snr, 'UTT']))
            video_root = os.path.join(feat_root, args.video_feature[-4].join([args.video_feature[:-4], args.snr, 'UTT']))
        print (f'audio feature root: {audio_root}')

        # --------------- temporal test ---------------
        # for name in names: assert os.path.exists(os.path.join(audio_root, name+'.npy'))

        # analyze params
        self.feat_type = args.feat_type
        self.feat_scale = args.feat_scale # 特征预压缩
        assert self.feat_scale >= 1
        assert self.feat_type in ['utt', 'frm_align', 'frm_unalign']

        # per-modality video scale override: 0/unset -> use feat_scale.
        # video is the listener's only signal (the prediction target); a lower
        # scale keeps more temporal detail (micro-expressions) than the global
        # feat_scale, without changing the speaker (audio/text) resolution.
        video_scale = getattr(args, 'video_feat_scale', 0) or self.feat_scale
        self.video_scale = video_scale
        assert video_scale >= 1

        # read datas (reduce __getitem__ durations)
        # Compress and truncate inside each worker to keep IPC payload small.
        # max_seqlen caps outlier-length sequences after scale compression.
        audios, self.adim = func_read_multiprocess(audio_root, self.names, read_type='feat', scale_factor=self.feat_scale)
        texts,  self.tdim = func_read_multiprocess(text_root,  self.names, read_type='feat', scale_factor=self.feat_scale)
        videos, self.vdim = func_read_multiprocess(video_root, self.names, read_type='feat', scale_factor=video_scale)

        ## read batch (reduce collater durations)
        # step2: align to batch
        if self.feat_type == 'utt': # -> 每个样本每个模态的特征压缩到句子级别
            audios, texts, videos = align_to_utt(audios, texts, videos)
        elif self.feat_type == 'frm_align':
            audios, texts, videos = align_to_text(audios, texts, videos) # 模态级别对齐
            audios, texts, videos = pad_to_maxlen_pre_modality(audios, texts, videos) # 样本级别对齐
        elif self.feat_type == 'frm_unalign':
            audios, texts, videos = pad_to_maxlen_pre_modality(audios, texts, videos) # 样本级别对齐
        self.audios, self.texts, self.videos = audios, texts, videos

        # optional 4th modality: listener body language (pose + optical flow).
        # only read when --body_feature is set; uses the video (listener) scale
        # so it keeps the same temporal resolution as the face-CLIP video.
        self.body_feature = getattr(args, 'body_feature', None)
        if self.body_feature:
            body_root = os.path.join(feat_root, self.body_feature)
            bodys, self.bdim = func_read_multiprocess(body_root, self.names, read_type='feat', scale_factor=video_scale)
            if self.feat_type == 'utt':
                bodys = [np.mean(b, axis=0) for b in bodys]
            else:  # frm_align / frm_unalign: pad to this modality's own max length
                body_maxlen = max(len(f) for f in bodys)
                bodys = [func_mapping_feature(b, body_maxlen) for b in bodys]
            self.bodys = bodys
        else:
            self.bodys, self.bdim = None, 0

 
    def __len__(self):
        return len(self.names)


    def __getitem__(self, index):
        instance = dict(
            audio = self.audios[index],
            text  = self.texts[index],
            video = self.videos[index],
            emo   = self.labels[index]['emo'],
            val   = self.labels[index]['val'],
            name  = self.names[index],
        )
        if self.bodys is not None:
            instance['body'] = self.bodys[index]
        return instance
    

    def collater(self, instances):
        audios = [instance['audio'] for instance in instances]
        texts  = [instance['text']  for instance in instances]
        videos = [instance['video'] for instance in instances]

        batch = dict(
            audios = torch.FloatTensor(np.array(audios)),
            texts  = torch.FloatTensor(np.array(texts)),
            videos = torch.FloatTensor(np.array(videos)),
        )
        if 'body' in instances[0]:
            bodys = [instance['body'] for instance in instances]
            batch['bodys'] = torch.FloatTensor(np.array(bodys))

        emos  = torch.LongTensor([instance['emo']  for instance in instances])
        vals  = torch.FloatTensor([instance['val']  for instance in instances])
        names = [instance['name'] for instance in instances]

        return batch, emos, vals, names
    

    def get_featdim(self):
        print (f'audio dimension: {self.adim}; text dimension: {self.tdim}; video dimension: {self.vdim}')
        return self.adim, self.tdim, self.vdim
    