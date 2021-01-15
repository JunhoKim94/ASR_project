class Config:
    def __init__(self, char_list):
        self.transformer_encoder_selfattn_layer_type = "selfattn"
        self.adim = 512
        self.aheads = 8
        self.wshare = 4
        self.ldconv_encoder_kernel_length = 11
        self.ldconv_usebias = False
        self.eunits = 2048
        self.elayers = 6
        self.transformer_input_layer = "conv2d"
        self.dropout_rate = 0.1
        self.transformer_attn_dropout_rate = 0.0
        self.ldconv_decoder_kernel_length = 11
        self.dunits = 2048
        self.dlayers = 6
        self.lsm_weight = 0.0
        self.transformer_length_normalized_loss = True
        self.transformer_decoder_selfattn_layer_type = "selfattn"
        self.ctc_type = "builtin"
        self.char_list = char_list
        self.sym_space = " "
        self.sym_blank = "-"
        self.report_cer = True
        self.report_wer = True
        self.mtlalpha = 0.3
        self.transformer_init = "pytorch"
        self.ignore_id = -1
        self.normalize = True
        self.specaug = True

class Recog_config:
    def __init__(self):
        self.ctc_weight = 0.0 #0.141849
        self.beam_size = 5
        self.penalty = 0.2 #0.592332
        self.maxlenratio = 0.8
        self.minlenratio = 0
        self.lm_weight = 0
        self.nbest = 1