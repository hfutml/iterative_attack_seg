'''
图片加载方式由直接读取原图修改为加载npy文件
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import h5py
import lmdb
import os
import numpy as np
import numpy.random as npr
import random

import torch
import torch.utils.data as data

import multiprocessing
import six
from skimage import io
from skimage.transform import resize
from PIL import Image
class HybridLoader:
    """
    If db_path is a director, then use normal file loading
    If lmdb, then load from lmdb
    The loading method depend on extention.

    in_memory: if in_memory is True, we save all the features in memory
               For individual np(y|z)s, we don't need to do that because the system will do this for us.
               Should be useful for lmdb or h5.
               (Copied this idea from vilbert)
    """

    def __init__(self, db_path, ext, in_memory=False):
        self.db_path = db_path
        self.ext = ext
        if self.ext == '.npy':
            self.loader = lambda x: np.load(six.BytesIO(x))
        else:
            def load_npz(x):
                x = np.load(six.BytesIO(x))
                return x['arr_0'] if 'arr_0' in x else x[
                    'z']  # normally it should be 'feat', but under cocotest_bu, the key is saved to be 'z' mistakenly.

            self.loader = load_npz
        if db_path.endswith('.lmdb'):
            # print('lmdbbbbbbbbbbbbbbbb')
            self.db_type = 'lmdb'
            self.env = lmdb.open(db_path, subdir=os.path.isdir(db_path),
                                 readonly=True, lock=False,
                                 readahead=False, meminit=False)
        elif db_path.endswith('.pth'):  # Assume a key,value dictionary
            self.db_type = 'pth'
            self.feat_file = torch.load(db_path)
            self.loader = lambda x: x
            print('HybridLoader: ext is ignored')
        elif db_path.endswith('h5'):
            # print('h55555555555555555')
            self.db_type = 'h5'
            self.loader = lambda x: np.array(x).astype('float32')
        else:
            print('dirrrrrrrrrrrrrrrrrrrrrrr')
            self.db_type = 'dir'

        self.in_memory = in_memory
        if self.in_memory:
            self.features = {}

    def get(self, key):
        # print('key: ', key)

        if self.in_memory and key in self.features:
            # We save f_input because we want to save the
            # compressed bytes to save memory
            f_input = self.features[key]
        elif self.db_type == 'lmdb':
            env = self.env
            with env.begin(write=False) as txn:
                byteflow = txn.get(key.encode())
            f_input = byteflow
        elif self.db_type == 'pth':
            print()
            f_input = self.feat_file[key]
        elif self.db_type == 'h5':
            f_input = h5py.File(self.db_path, 'r')[key]
        else:
            # print('os_path: ', os.path.join(self.db_path, key + self.ext))
            f_input = open(os.path.join(self.db_path, key + self.ext), 'rb').read()
            # print('f_input: ', f_input)

        if self.in_memory and key not in self.features:
            self.features[key] = f_input

        # load image
        feat = self.loader(f_input)
        # print('feat: ', feat)

        return feat


class Dataset(data.Dataset):

    def get_vocab_size(self):
        return self.vocab_size

    def get_vocab(self):
        return self.ix_to_word

    def get_seq_length(self):
        return self.seq_length

    def get_src_vocab_size(self):
        return self.src_vocab_size

    def get_src_vocab(self):
        return self.src_ix_to_word

    def __init__(self, opt):
        self.opt = opt
        # self.seq_per_img = opt.seq_per_img  # 1 一张照片的captions数量
        # print('self.seq_per_img: ', self.seq_per_img)
        self.seq_per_img = 1

        # feature related options
        self.norm_att_feat = getattr(opt, 'norm_att_feat', 0)
        self.story_text = json.load(open(self.opt.input_text_json))

        # load the json file which contains additional information about the dataset
        print('DataLoader loading json file: ', opt.input_json)
        self.info = json.load(open(self.opt.input_json))

        # self.test_images = np.load(self.opt.test_input_images_npy)
        # self.test_images_dict = json.load(open(self.opt.test_images_dict_json))
        if 'tgt_ix_to_word' in self.info:
            self.ix_to_word = self.info['tgt_ix_to_word']
            self.vocab_size = len(self.ix_to_word)
            print('tgt vocab size is ', self.vocab_size)

        self.src_ix_to_word = self.info['src_ix_to_word']
        self.src_vocab_size = len(self.src_ix_to_word)
        print('src vocab size is', self.src_vocab_size)
        self.src_word_to_ix = self.info['src_word_to_ix']
        # np.save('src_word_to_ix.npy', self.src_word_to_ix)


        # open the hdf5 file
        print('DataLoader loading h5 file: ', opt.input_fc_dir, opt.input_conv_dir, opt.input_box_dir,
              opt.input_label_h5)
        """
        Setting input_label_h5 to none is used when only doing generation.
        For example, when you need to test on coco test set.
        """
        if self.opt.input_label_h5 != 'none':
            self.h5_label_file = h5py.File(self.opt.input_label_h5, 'r', driver='core')
            # load in the sequence data
            seq_size = self.h5_label_file['labels'].shape
            self.label = self.h5_label_file['labels'][:]
            self.seq_length = seq_size[1]
            print('max sequence length in data is', self.seq_length)
            # load the pointers in full to RAM (should be small enough)
            self.label_start_ix = self.h5_label_file['label_start_ix'][:]
            self.label_end_ix = self.h5_label_file['label_end_ix'][:]
        else:
            self.seq_length = 1

        # download src adj
        self.src = h5py.File(self.opt.input_src_h5, 'r', driver='core')
        self.sent1 = self.src['sent1'][:]
        self.sent2 = self.src['sent2'][:]
        self.sent3 = self.src['sent3'][:]
        self.sent4 = self.src['sent4'][:]

        self.adj = h5py.File(self.opt.input_adj_h5, 'r', driver="core")
        self.adj1 = self.adj['adj1'][:]
        self.adj2 = self.adj['adj2'][:]
        self.adj3 = self.adj['adj3'][:]
        self.adj4 = self.adj['adj4'][:]

        self.data_in_memory = getattr(opt, 'data_in_memory', False)
        # self.fc_loader = HybridLoader(self.opt.input_fc_dir, '.npy', in_memory=self.data_in_memory)
        # self.att_loader = HybridLoader(self.opt.input_att_dir, '.npz', in_memory=self.data_in_memory)
        # self.box_loader = HybridLoader(self.opt.input_box_dir, '.npy', in_memory=self.data_in_memory)

        # self.images_loader =  HybridLoader(self.opt.val_input_images_file, '.npy', in_memory=self.data_in_memory)

        self.num_images = len(self.info['images'])  # self.label_start_ix.shape[0]
        print('read %d image features' % (self.num_images))

        # separate out indexes for each of the provided splits
        self.split_ix = {'train': [], 'val': [], 'test': []}
        self.src_story_id = {}
        self.imgId_2_ix = {}
        self.imgIx_2id = {}
        for ix in range(len(self.info['images'])):
            img = self.info['images'][ix]
            self.src_story_id[ix] = self.info['images'][ix]['story_id']
            self.imgId_2_ix[self.info['images'][ix]['id']] = ix
            if not 'split' in img:
                self.split_ix['train'].append(ix)
                self.split_ix['val'].append(ix)
                self.split_ix['test'].append(ix)
            elif img['split'] == 'train':
                self.split_ix['train'].append(ix)
                # self.imgIx_2id[ix] = self.info['images'][ix]['id']
            elif img['split'] == 'val':
                self.split_ix['val'].append(ix)
                # self.imgIx_2id[ix] = self.info['images'][ix]['id']
            elif img['split'] == 'test':
                self.split_ix['test'].append(ix)
                # self.imgIx_2id[ix] = self.info['images'][ix]['id']
            elif opt.train_only == 0:  # restval
                self.split_ix['train'].append(ix)



        print('assigned %d images to split train' % len(self.split_ix['train']))
        print('assigned %d images to split val' % len(self.split_ix['val']))
        print('assigned %d images to split test' % len(self.split_ix['test']))


        with open('/data/home/wangyouze/projects/others/MMT/data/VIST-E/story_id_dict.json', 'w') as ff:
            json.dump(self.src_story_id, ff)
        ff.close()

    def get_captions(self, ix, seq_per_img):
        # fetch the sequence labels
        ix1 = self.label_start_ix[ix] - 1  # label_start_ix starts from 1
        ix2 = self.label_end_ix[ix] - 1
        ncap = ix2 - ix1 + 1  # number of captions available for this image
        assert ncap > 0, 'an image does not have any label. this can be handled but right now isn\'t'

        if ncap < seq_per_img:
            # we need to subsample (with replacement)
            seq = np.zeros([seq_per_img, self.seq_length], dtype='int')
            for q in range(seq_per_img):
                ixl = random.randint(ix1, ix2)
                seq[q, :] = self.label[ixl, :self.seq_length]
        else:
            ixl = random.randint(ix1, ix2 - seq_per_img + 1)
            seq = self.label[ixl: ixl + seq_per_img, :self.seq_length]

        return seq

    def get_src_text(self, ix, sent):
        if sent == "sent1":
            src = self.story_text[ix][0]
        elif sent == "sent2":
            src = self.story_text[ix][1]
        elif sent == "sent3":
            src = self.story_text[ix][2]
        elif sent == "sent4":
            src = self.story_text[ix][3]
        return src


    def collate_func(self, batch, split):
        seq_per_img = self.seq_per_img
        img_batch = []
        story_id_batch = []
        label_batch = []
        src1_batch = []
        src2_batch = []
        src3_batch = []
        src4_batch = []

        story_end_batch = []

        gts = []

        for sample in batch:
            story_id, tmp_img, tmp_seq, story_end, \
            ix, it_pos_now, tmp_wrapped, \
            tem_src1, tem_src2, tem_src3, tem_src4 = sample
            if tmp_wrapped:
                wrapped = True

            story_id_batch.append(story_id)
            img_batch.append(tmp_img)
            story_end_batch.append(story_end)

            src1_batch.append(tem_src1)
            src2_batch.append(tem_src2)
            src3_batch.append(tem_src3)
            src4_batch.append(tem_src4)

            tmp_label = np.zeros([seq_per_img, self.seq_length + 2], dtype='int')
            if hasattr(self, 'h5_label_file'):
                # if there is ground truth
                tmp_label[:, 1: self.seq_length + 1] = tmp_seq
                # print('tmp_label_size: ', tmp_label.shape)  # (5, 22)
            label_batch.append(tmp_label)

            # Used for reward evaluation
            if hasattr(self, 'h5_label_file'):
                # if there is ground truth
                gts.append(self.label[self.label_start_ix[ix] - 1: self.label_end_ix[ix]])  # shape(5, 20)
                # print('gts_shape', (self.label[self.label_start_ix[ix] - 1: self.label_end_ix[ix]]).shape)
            else:
                gts.append([])


        # #sort by att_feat length
        # fc_batch, att_batch, label_batch, gts, infos = \
        #     zip(*sorted(zip(fc_batch, att_batch, np.vsplit(label_batch, batch_size), gts, infos), key=lambda x: len(x[1]), reverse=True))
        story_id_batch, img_batch, label_batch, gts, \
        src1_batch, src2_batch, src3_batch, src4_batch = \
            zip(*sorted(zip(story_id_batch, img_batch, label_batch, gts,
                            src1_batch, src2_batch, src3_batch, src4_batch), key=lambda x: 0, reverse=True))

        data = {}
        data['src1'] = np.stack(src1_batch)

        data['src2'] = np.stack(src2_batch)

        data['src3'] = np.stack(src3_batch)

        data['src4'] = np.stack(src4_batch)


        # np.stack的官方解释为 对指定axis增加维度  由（2048，）变为(batch, 2048)
        # print('data[fc_feats]_size: ', data['fc_feats'].shape)
        # merge att_feats
        data['story_id'] = np.stack(story_id_batch)
        data['image'] = np.stack(img_batch)
        data['labels'] = np.vstack(label_batch)
        data['story_end'] = np.stack(story_end_batch)

        nonzeros = np.array(list(map(lambda x: (x != 0).sum() + 2, data['labels'])))
        mask_batch = np.zeros([data['labels'].shape[0], self.seq_length + 2], dtype='float32')
        for ix, row in enumerate(mask_batch):
            row[:nonzeros[ix]] = 1
        data['masks'] = mask_batch
        data['labels'] = data['labels'].reshape(len(batch), seq_per_img, -1)
        data['masks'] = data['masks'].reshape(len(batch), seq_per_img, -1)

        data['gts'] = gts  # all ground truth captions of each images

        # data = {k: torch.from_numpy(v) if type(v) is np.ndarray and k != "file_path" and k != "target" and k != "story_id" and k !="" else v for k, v in data.items()}

        return data
    def get_target_data(self, index):
        ix, it_pos_now, wrapped = index
        images_feat_id = self.info['images'][ix]['id']

        if hasattr(self, 'h5_label_file'):
            # print('has h5_label_file attribute')
            # hasattr() 函数用于判断对象是否包含对应的属性
            seq = self.get_captions(ix, self.seq_per_img).astype(int)
        else:
            seq = None

        src1 = np.stack([self.get_src_text(ix, 'sent1')])
        src2 = np.stack([self.get_src_text(ix, 'sent2')])
        src3 = np.stack([self.get_src_text(ix, 'sent3')])
        src4 = np.stack([self.get_src_text(ix, 'sent4')])

        res = {'file_path':images_feat_id, 'labels':seq,
                'src1':src1, 'src2':src2, 'src3':src3, 'src4':src4}
        res = {k: torch.from_numpy(v) if k == 'labels' else v for k, v in res.items()}
        # res = {k: torch.from_numpy(v) if type(v) is np.ndarray and k != 'file_path' else v for k, v in res.items()}
        return res

    def __getitem__(self, index):
        """This function returns a tuple that is further passed to collate_fn
        """
        # print('index: ', index)
        ix, it_pos_now, wrapped = index  # self.split_ix[index]

        img_id = str(self.info['images'][ix]['id'])
        jpg_img_path = '/data/home/wangyouze/dataset/VIST-E/test_images/test/' + img_id + '.jpg'
        png_img_path = '/data/home/wangyouze/dataset/VIST-E/test_images/test/' + img_id + '.png'
        if os.path.exists(jpg_img_path):
            # print(jpg_img_path)
            img = io.imread(jpg_img_path)
            img = resize(img, (224, 224))
        # elif os.path.exists(png_img_path):
        #     if os.path.exists('/mfs/wenbo/VIST/images/test_images/c4_2_c3/' + img_id + '_png' + '.png'):
        #         img = io.imread('/mfs/wenbo/VIST/images/test_images/c4_2_c3/' + img_id + '_png' + '.png')
        #         img = resize(img, (224, 224))
        #     else:
        #         img = Image.open(png_img_path)
        #         img = img.convert("RGB")
        #         img.save('/mfs/wenbo/VIST/images/test_images/c4_2_c3' + '/' +img_id + '_png' + '.png')
        #         img = io.imread(
        #             '/mfs/wenbo/VIST/images/test_images/c4_2_c3' + '/' +img_id + '_png' + '.png')

        #         # img = io.imread(png_img_path)
        #         img = resize(img, (224, 224))
        else:
            img = np.zeros((224, 224, 3), dtype=np.float_)
            print('Not found image!!!')

        story_id = self.info['images'][ix]['story_id']
        story_end = self.info['images'][ix]['story_end']
        if story_id== '45597':
            print('123')
        if hasattr(self, 'h5_label_file'):
            # print('has h5_label_file attribute')
            # hasattr() 函数用于判断对象是否包含对应的属性
            seq = self.get_captions(ix, self.seq_per_img)
        else:
            seq = None

        # src, adj

        # print('ix: ', ix)
        src1 = self.get_src_text(ix, 'sent1')
        src2 = self.get_src_text(ix, 'sent2')
        src3 = self.get_src_text(ix, 'sent3')
        src4 = self.get_src_text(ix, 'sent4')


        return (story_id, img, seq, story_end,
                ix, it_pos_now, wrapped,
                src1, src2, src3, src4)

    def __len__(self):
        return len(self.info['images'])


class DataLoader:
    def __init__(self, opt, split='test', shuffle=False):
        self.opt = opt
        self.batch_size = self.opt.batch_size
        self.dataset = Dataset(opt)

        # Initialize loaders and iters
        self.loaders, self.iters = {}, {}

        sampler = MySampler(self.dataset.split_ix[split], shuffle=False, wrap=False)
        self.loaders[split] = data.DataLoader(dataset=self.dataset,
                                              batch_size=self.batch_size,
                                              sampler=sampler,
                                              pin_memory=True,
                                              num_workers=0,  # 4 is usually enough
                                              collate_fn=lambda x: self.dataset.collate_func(x, split),
                                              drop_last=False,
                                              shuffle=shuffle)

        # sampler 定义取batch的方法，是一个迭代器， 每次生成一个key 用于读取dataset中的值
        # collate_fn：如何取样本的，我们可以定义自己的函数来准确地实现想要的功能
        self.iters[split] = iter(self.loaders[split])

    def get_batch(self, split):
        try:
            data = next(self.iters[split])
        except StopIteration:
            self.iters[split] = iter(self.loaders[split])
            data = next(self.iters[split])
        return data

    def reset_iterator(self, split):
        self.loaders[split].sampler._reset_iter()
        self.iters[split] = iter(self.loaders[split])

    def get_vocab_size(self):
        return self.dataset.get_vocab_size()

    def get_src_vocab_size(self):
        return self.dataset.get_src_vocab_size()

    # @property
    # @property 1.修饰方法，是方法可以像属性一样访问 2.与所定义的属性配合使用，这样可以防止属性被修改

    @property
    def vocab_size(self):
        return self.get_vocab_size()

    @property
    def src_vocab_size(self):
        return self.get_src_vocab_size()

    def get_vocab(self):
        return self.dataset.get_vocab()

    def get_src_vocab(self):
        return self.dataset.get_src_vocab()

    def get_seq_length(self):
        return self.dataset.get_seq_length()

    @property
    def seq_length(self):
        return self.get_seq_length()

    def state_dict(self):
        def get_prefetch_num(split):
            if self.loaders[split].num_workers > 0:
                return (self.iters[split]._send_idx - self.iters[split]._rcvd_idx) * self.batch_size
            else:
                return 0

        return {split: loader.sampler.state_dict(get_prefetch_num(split)) \
                for split, loader in self.loaders.items()}

    def load_state_dict(self, state_dict=None):
        if state_dict is None:
            return
        for split in self.loaders.keys():
            self.loaders[split].sampler.load_state_dict(state_dict[split])


class MySampler(data.sampler.Sampler):
    def __init__(self, index_list, shuffle, wrap):
        self.index_list = index_list
        self.shuffle = shuffle
        self.wrap = wrap
        # if wrap, there will be not stop iteration called
        # wrap True used during training, and wrap False used during test.
        self._reset_iter()

    def __iter__(self):
        return self

    def __next__(self):
        wrapped = False
        if self.iter_counter == len(self._index_list):
            self._reset_iter()
            if self.wrap:
                wrapped = True
            else:
                raise StopIteration()
        if len(self._index_list) == 0:  # overflow when 0 samples
            return None
        elem = (self._index_list[self.iter_counter], self.iter_counter + 1, wrapped)
        self.iter_counter += 1
        return elem

    def next(self):
        return self.__next__()

    def _reset_iter(self):
        if self.shuffle:
            rand_perm = npr.permutation(len(self.index_list))
            self._index_list = [self.index_list[_] for _ in rand_perm]
        else:
            self._index_list = self.index_list

        self.iter_counter = 0

    def __len__(self):
        return len(self.index_list)

    def load_state_dict(self, state_dict=None):
        if state_dict is None:
            return
        self._index_list = state_dict['index_list']
        self.iter_counter = state_dict['iter_counter']

    def state_dict(self, prefetched_num=None):
        prefetched_num = prefetched_num or 0
        return {
            'index_list': self._index_list,
            'iter_counter': self.iter_counter - prefetched_num
        }