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




class ASRModel(ESPnetASRModel):
    def __init__(
        self,
        vocab_size: int,
        token_list: Union[Tuple[str, ...], List[str]],
        frontend: Optional[AbsFrontend],
        specaug: Optional[AbsSpecAug],
        normalize: Optional[AbsNormalize],
        encoder: AbsEncoder,
        decoder: AbsDecoder,
        ctc: CTC,
        rnnt_decoder: None,
        ctc_weight: float = 0.5,
        ignore_id: int = -1,
        lsm_weight: float = 0.0,
        length_normalized_loss: bool = False,
        report_cer: bool = True,
        report_wer: bool = True,
        sym_space: str = "<space>",
        sym_blank: str = "<blank>",
    ):
        super(ASRModel, self).__init__(vocab_size, 
                                    token_list, 
                                    frontend, 
                                    specaug, 
                                    normalize, 
                                    encoder, 
                                    decoder, 
                                    ctc, 
                                    rnnt_decoder, 
                                    ctc_weight, 
                                    ignore_id, 
                                    lsm_weight, 
                                    length_normalized_loss, 
                                    report_cer, 
                                    report_wer, 
                                    sym_space, 
                                    sym_blank)

    def recognize(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor
        ):
        
        """
        Frontend + Encoder. Note that this method is used by asr_inference.py

        Args:
            speech: (Batch, Length, ...)
            speech_lengths: (Batch, )
        """

        encoder_out, encoder_out_lens = self.encode(speech, speech_lengths)
        
        ys_in_pad = torch.zeros((speech.shape[0], 1)).fill_(self.sos)
        # 1. Forward decoder
        state = None
        for i in range(20):
            decoder_out, state = self.decoder.batch_score(
                ys_in_pad, state, encoder_out
            )
            decoder_out = torch.topk(decoder_out, k = 1, dim = -1)[0]
            ys_in_pad = torch.cat([ys_in_pad, decoder_out], dim = 1)

        return ys_in_pad

