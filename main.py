from espnet.nets.pytorch_backend.e2e_asr import E2E
from espnet2.asr.espnet_joint_model import ESPnetEnhASRModel
from espnet2.asr.espnet_model import ESPnetASRModel
from espnet2.asr.decoder.transformer_decoder import TransformerDecoder
from espnet2.asr.encoder.transformer_encoder import TransformerEncoder
from espnet2.asr.ctc import CTC
from espnet2.asr.specaug.specaug import SpecAug
from espnet2.asr.frontend.default import DefaultFrontend

from model.frontend import CustomFrontend

input_size = 100
vocab_size = 50
'''
vocab_size: int,
token_list: Union[Tuple[str, ...], List[str]],
enh: Optional[AbsEnhancement],
frontend: Optional[AbsFrontend],
specaug: Optional[AbsSpecAug],
normalize: Optional[AbsNormalize],
encoder: AbsEncoder,
decoder: AbsDecoder,
ctc: CTC,
'''
#Model 선언을 위해 필요한 모듈들
#vocab size(character 와 token 단위 중 선택해야 함.)
vocab_size = 4
#token list(해당 corpus의 token들 나열)
token_list = ["가","나", "<blank>", "<space>"]
#enh = None
#전처리 과정을 담당하는 class
frontend = DefaultFrontend()
#Data augmentation을 담당하는 class --> 시간적 비용이 많을 시 생략할것
specaug = SpecAug()
#전처리 후 데이터 normalize를 담당하는 class
normalize = None
#Transformer (Seq2Seq) 모델 --> 우리 Acoustic 모델
encoder =  TransformerEncoder(input_size)
decoder = TransformerDecoder(input_size, encoder_output_size = 256)
#CTC loss added
ctc = CTC(input_size, 256)
#전체 model binding
model = ESPnetASRModel(vocab_size = vocab_size, 
                            encoder = encoder, 
                            decoder = decoder, 
                            ctc = ctc, 
                            token_list = token_list,  
                            frontend = frontend, 
                            specaug = specaug, 
                            normalize = normalize,
                            rnnt_decoder = None)

print(model)