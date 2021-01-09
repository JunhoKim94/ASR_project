from espnet.nets.pytorch_backend.transformer.add_sos_eos import add_sos_eos
from contextlib import contextmanager
from distutils.version import LooseVersion
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import torch
from typeguard import check_argument_types

from espnet.nets.e2e_asr_common import ErrorCalculator
from espnet.nets.pytorch_backend.nets_utils import th_accuracy
from espnet.nets.pytorch_backend.transformer.add_sos_eos import add_sos_eos
from espnet.nets.pytorch_backend.transformer.label_smoothing_loss import (
    LabelSmoothingLoss,  # noqa: H301
)
from espnet2.asr.ctc import CTC
from espnet2.asr.decoder.abs_decoder import AbsDecoder
from espnet2.asr.encoder.abs_encoder import AbsEncoder
from espnet2.asr.frontend.abs_frontend import AbsFrontend
from espnet2.asr.specaug.abs_specaug import AbsSpecAug
from espnet2.asr.specaug.specaug import SpecAug
from espnet2.layers.abs_normalize import AbsNormalize
from espnet2.layers.utterance_mvn import UtteranceMVN
from espnet2.torch_utils.device_funcs import force_gatherable
from espnet2.train.abs_espnet_model import AbsESPnetModel
#from espnet.nets.pytorch_backend.e2e_asr_transformer import E2E
from model.e2e import E2E
from model.frontend import CustomFrontend
from espnet2.asr.specaug.specaug import SpecAug

SAMPLE_RATE = 16000

class ASRModel(torch.nn.Module):
    def __init__(
        self,
        input_size: int,
        vocab_size: int,
        token_list: Union[Tuple[str, ...], List[str]],
        device,
        config
    ):
        super().__init__()

        self.sos = vocab_size - 1
        self.eos = vocab_size - 1
        self.vocab_size = vocab_size
        self.ignore_id = config.ignore_id
        self.ctc_weight = config.mtlalpha
        self.token_list = config.char_list.copy()


        self.specaug = SpecAug() if config.specaug else None
        self.normalize = UtteranceMVN() if config.normalize else None

        self.frontend = CustomFrontend(fs = SAMPLE_RATE,
                            n_fft= 512,
                            normalized = True,
                            hop_length= int(0.01 * SAMPLE_RATE),
                            win_length= int(0.03 * SAMPLE_RATE), 
                            n_mels = 80)


        self.model = E2E(input_size, vocab_size, config, device)


    def _extract_feats(
        self, speech: torch.Tensor, speech_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert speech_lengths.dim() == 1, speech_lengths.shape

        # for data-parallel
        speech = speech[:, : speech_lengths.max()]

        if self.frontend is not None:
            # Frontend
            #  e.g. STFT and Feature extract
            #       data_loader may send time-domain signal in this case
            # speech (Batch, NSamples) -> feats: (Batch, NFrames, Dim)
            feats, feats_lengths = self.frontend(speech, speech_lengths)
        else:
            # No frontend and no feature extract
            feats, feats_lengths = speech, speech_lengths

        # 2. Data augmentation for spectrogram
        if self.specaug is not None and self.training:
            feats, feats_lengths = self.specaug(feats, feats_lengths)

        # 3. Normalization for feature: e.g. Global-CMVN, Utterance-CMVN
        if self.normalize is not None:
            feats, feats_lengths = self.normalize(feats, feats_lengths)


        return feats, feats_lengths

    def forward(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:

        feats, feats_length = self._extract_feats(speech, speech_lengths)
        loss, ret_dict = self.model(feats, feats_length, text)
        #acc = self.model.acc

        return loss, ret_dict
    
    def recognize(self, speech, speech_lengths, recog_args, char_list=None, rnnlm=None, use_jit=False):

        feats, feats_length = self._extract_feats(speech, speech_lengths)
        #feats = feats.squeeze(0).detach().cpu().numpy()
        results = []
        for feat in feats:
            hypo = self.model.recognize(feat, recog_args, char_list =  self.token_list)
            results.append(hypo[0]["yseq"])
            
        return results