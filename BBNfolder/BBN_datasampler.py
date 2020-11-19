import torch
from torch.utils.data import Dataset
import os
import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt
from ReadTXT import readCOP, readCOP_cat, readCOP_cat_unrepeated
from ReadExcel import get_sot, get_score, get_cal, get_cvemp, get_ovemp, get_svv, get_rot
from tqdm import tqdm
import termcolor

from RecurrencePlot import rec_plot
from utils import shift_signal, reshape_signal, normalize_by_height, expand_data_randomly


def toBlue(str): return termcolor.colored(str, "blue", attrs=["bold"])


def gen_filename(raw_names, filenames):
    new_names = []
    for rn in raw_names:
        rn = str(int(rn))
        new_names.append(list(filter(lambda x: x.split(' ')[0][-len(rn):] == rn, filenames))[0])
    return new_names


def makeSpec(signal, nperseg):
    _, _, spec = sig.spectrogram(x=signal, fs=100, nperseg=nperseg, nfft=nperseg, noverlap=int(nperseg * 0.5))
    spec = 10 * np.log10(spec, out=np.zeros_like(spec), where=(spec != 0))
    return torch.from_numpy(spec)


class COP_Data(Dataset):
    def __init__(self, valid_info, xl_sheet, mode='spec',
                 key='COG', sot='all', root="/home/wine1865/다운로드/Ansan/data/COP Data/", nperseg=100, rp_steps=10,
                 split=2000, normalize=False, sampling=None, scale_factor=1, emb=False, eps=None, score=False,
                 val_reg=False, shuffle_trial=False, gt='sot', expand_data=0, normalize_rp=None,
                 class_ratio=None):
        self.split = split
        self.root = root
        indexes, names = [i for i, j in valid_info], [j for i, j in valid_info]
        files = list(filter(lambda x: x[-3:] == 'txt', os.listdir(self.root)))
        self.files = gen_filename(names, files)
        # signals before conversion to spectrogram
        self.signals_x = []
        # self.signals_y = []
        # corresponding labels
        self.labels = []
        # excel sheet
        self.st = xl_sheet
        self.mode = mode
        self.sot = sot
        self.key = key
        self.normalize = normalize
        self.sampling = sampling
        self.scale_factor = scale_factor
        self.emb = emb
        self.num = []
        self.score = score
        self.val_reg = val_reg
        self.shuffle_trial = shuffle_trial
        self.gt = gt
        self.expand_data = expand_data
        self.normalize_rp = normalize_rp
        if class_ratio is None:
            self.class_ratio = (1, 1)
        else:
            self.class_ratio = class_ratio
        if eps is None:
            self.eps = 0.01
        else:
            self.eps = eps
        if self.mode == 'rec':
            self.rp_steps = rp_steps
        elif self.mode == 'spec':
            self.nperseg = nperseg
        print(toBlue("Preparing data"))
        bar = tqdm(total=len(self.files))
        for i, file in enumerate(self.files):
            file = os.path.join(self.root, file)
            # append signals(length of each is 2000)
            if self.emb:
                sig = readCOP_cat_unrepeated(file, SOT=self.sot,
                                                  key=self.key, split=self.split, sheet=self.st)
                # sig = readCOP_cat(file, SOT=self.sot, key=self.key, split=self.split, sheet=self.st,
                #                   shuffle_trial=self.shuffle_trial)
                # self.num.append(num)
            else:
                if type(self.sot) == str:
                    sig = readCOP(file, SOT=self.sot, key=self.key, split=self.split, sheet=self.st)
                elif type(self.sot) == list:
                    sig = readCOP_cat(file, SOT=self.sot, key=self.key, split=self.split, sheet=self.st,
                                      shuffle_trial=self.shuffle_trial)
                else:
                    raise Exception("Wrong input param for SOT")
            # sig = shift_signal(sig)
            if self.normalize:
                self.signals_x.extend(normalize_by_height(file, sig))
            else:
                self.signals_x.extend(sig)
            # self.signals_y.extend(sig[1])
            if self.score:
                if self.gt == 'sot':
                    label = np.asarray(get_score(self.st, indexes[i]))
                elif self.gt == 'cal':
                    label = np.asarray(get_cal(self.st, indexes[i]))
            else:
                if self.gt == 'sot':
                    label = np.asarray(get_sot(self.st, indexes[i]))
                elif self.gt == 'cal':
                    label = np.asarray(get_cal(self.st, indexes[i]))
                elif self.gt == 'cvemp':
                    label = np.asarray(get_cvemp(self.st, indexes[i]))
                elif self.gt == 'ovemp':
                    label = np.asarray(get_ovemp(self.st, indexes[i]))
                elif self.gt == 'svv':
                    label = np.asarray(get_svv(self.st, indexes[i]))
                elif self.gt == 'rotary':
                    label = np.asarray(get_rot(self.st, indexes[i]))
                elif self.gt == 'multi':
                    label = np.asarray(get_cal(self.st, indexes[i]))
                    label = np.append(label, np.asarray(get_cvemp(self.st, indexes[i])))
                    label = np.append(label, np.asarray(get_ovemp(self.st, indexes[i])))
                    label = np.append(label, np.asarray(get_svv(self.st, indexes[i])))
                    label = np.append(label, np.asarray(get_rot(self.st, indexes[i])))
            for _ in range(len(sig)):
                self.labels.append(label)
            bar.update()
        bar.close()
        self.labels, self.signals_x = np.asarray(self.labels), np.asarray(self.signals_x)
        sort_ind = np.argsort(self.labels[:, 0])
        self.labels, self.signals_x = self.labels[sort_ind], self.signals_x[sort_ind]

    def __len__(self):
        return len(self.signals_x)

    def __getitem__(self, idx):
        if self.emb:
            # pass signals directly
            return self.signals_x[idx], self.labels[idx]
        if self.mode == 'spec':
            spec_list = []
            for sig in self.signals_x[idx]:
                if self.sampling is not None:
                    sig = reshape_signal(sig, self.sampling)
                spec_list.append(makeSpec(sig, self.nperseg))
            spec = torch.zeros(len(spec_list), spec_list[0].size()[0], spec_list[0].size()[1], dtype=spec_list[0].dtype)
            for i, s in enumerate(spec_list):
                spec[i, :, :] = s
            return spec, self.labels[idx]
        elif self.mode == 'rec':
            rec_list = []
            for sig in self.signals_x[idx]:
                if self.sampling is not None:
                    sig = reshape_signal(sig, self.sampling)
                if self.expand_data:
                    sig = expand_data_randomly(sig)
                # rp = np.squeeze(self.make_rec.transform(np.expand_dims(np.asarray(sig), 0)))
                rp = rec_plot(np.asarray(sig), eps=self.eps, steps=self.rp_steps)
                rp = torch.nn.functional.interpolate(
                    torch.from_numpy(rp).unsqueeze(0).unsqueeze(0),
                    scale_factor=self.scale_factor, mode='bilinear', align_corners=False).squeeze(0)
                if self.normalize_rp is not None:
                    rp = rp / self.normalize_rp
                rec_list.append(rp)
            rec = torch.zeros(len(rec_list), rec_list[0].size()[-2], rec_list[0].size()[-1], dtype=rec_list[0].dtype)
            for i, r in enumerate(rec_list):
                rec[i, :, :] = r.squeeze(0)
            if self.emb:
                # return rec, self.labels[idx], self.num[idx]
                return rec, self.labels[idx]
            else:
                if self.val_reg:
                    return rec, self.labels[idx], os.path.join(self.root, self.files[idx])
                else:
                    return rec, self.labels[idx]
        elif self.mode == 'none':
            sig_list = self.signals_x[idx]
            sig_tensor = torch.zeros((len(sig_list), 2000))
            for i, s in enumerate(sig_list):
                if s.shape[0] < 2000:
                    new_s = np.zeros(2000, dtype=np.float)
                    # compute n for repeating
                    n = int(2000 / s.shape[0])
                    for j in range(n):
                        new_s[:, s.shape[0] * j:s.shape[0] * (j + 1)] = s
                    # compute residual and fill it
                    residual = 2000 - n * s.shape[0]
                    if residual:
                        new_s[:, -residual:] = s[:, :residual]
                    s = new_s
                sig_tensor[i, :] = torch.from_numpy(s)
            return sig_tensor, self.labels[idx]
        else:
            raise Exception("Wrong mode")
