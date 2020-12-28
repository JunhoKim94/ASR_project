from espnet2.asr.espnet_model import ESPnetASRModel
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
from espnet2.layers.abs_normalize import AbsNormalize
from espnet2.torch_utils.device_funcs import force_gatherable
from espnet2.train.abs_espnet_model import AbsESPnetModel
from espnet.nets.pytorch_backend.e2e_asr_transformer import E2E



class ASRModel(torch.nn.Module):
    def __init__(
        self,
        input_size: int,
        vocab_size: int,
        token_list: Union[Tuple[str, ...], List[str]],
        frontend: Optional[AbsFrontend],
        specaug: Optional[AbsSpecAug],
        normalize: Optional[AbsNormalize],
        config
    ):
        super().__init__()

        self.sos = vocab_size - 1
        self.eos = vocab_size - 1
        self.vocab_size = vocab_size
        self.ignore_id = config.ignore_id
        self.ctc_weight = config.mtlalpha
        self.token_list = config.char_list.copy()

        self.frontend = frontend
        self.specaug = specaug
        self.normalize = normalize

        self.model = E2E(input_size, vocab_size, config)

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
            if self.adddiontal_utt_mvn is not None:
                feats, feats_lengths = self.adddiontal_utt_mvn(feats, feats_lengths)


        return feats, feats_lengths

    def forward(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:

        feats, feats_length = self._extract_feats(speech, speech_lengths)
        loss = self.model(feats, feats_length, text)

        return loss
    
    def recognize(self, speech, speech_lengths, recog_args, char_list=None, rnnlm=None, use_jit=False):

        feats, feats_length = self._extract_feats(speech, speech_lengths)
        #feats = feats.squeeze(0).detach().cpu().numpy()
        hypo = self.model.recognize(feats, recog_args, self.token_list)

        return hypo