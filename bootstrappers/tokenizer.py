import torch 
from transformers import AutoTokenizer,PreTrainedTokenizer,AutoConfig,AutoModel



def bootstrap_LM_tokenizer(model_name:str) -> PreTrainedTokenizer:
    model = AutoModel.from_pretrained(model_name,trust_remote_code=True)
    if hasattr(model, 'tokenizer') and model.tokenizer is not None:
        return model.tokenizer
    return AutoTokenizer.from_pretrained(model_name,trust_remote_code=True)

def bootstrap_convolution_tokenizer(model_name:str) -> None :
    return 

tokenizer_to_bootstrapper = {
    'LM':bootstrap_LM_tokenizer,
    'convolution': bootstrap_convolution_tokenizer,
}


def build_tokenizer_from_configuration(name: str,model_name:str) -> PreTrainedTokenizer:
    supported_optimizers = list(tokenizer_to_bootstrapper.keys())
    if name not in tokenizer_to_bootstrapper.keys():
        raise Exception(f'optimizer: {name} not supported. supported optimizers: {supported_optimizers}')
    return tokenizer_to_bootstrapper[name](model_name)