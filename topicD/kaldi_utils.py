import kaldi_io
import torch.utils.data as data


def parsecopyfeats(feat, cmvn=False, delta=False, splice=None):
    outstr = "copy-feats ark:{} ark:- |".format(feat)
    if cmvn:
        outstr += "apply-cmvn-sliding --norm-vars=false --cmn-window=300 --center ark:- ark:- |"
    if delta:
        outstr += "add-deltas ark:- ark:- |"
    if splice and splice > 0:
        outstr += "splice-feats --left-context={} --right-context={} ark:- ark:- |".format(
            splice, splice)
    return outstr


class KaldiDatasetEval(data.Dataset):

    def __init__(self, scp, stream, **kwargs):
        super(KaldiDatasetEval, self).__init__()
        # stream = parsecopyfeats(ark, kwargs)
        self.data_generator = kaldi_io.read_mat_ark(stream)
        self.scp = scp

    def __getitem__(self, idx):
        key, feat = next(self.data_generator)
        return feat, key

    def __len__(self):
        return len(open(self.scp).readlines())


