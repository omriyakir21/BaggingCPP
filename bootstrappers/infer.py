
from models.Convolution import CNNModel
from models.inference_LM import inference
from models.Convolution import CNNModel

def bootstrap_LM_infer(values_for_training_dict:dict) -> dict:  
    inference_results = inference(model=values_for_training_dict['model'],
                                dataset=values_for_training_dict['dataset'],
                                tokenizer=values_for_training_dict['tokenizer'],
                                device=values_for_training_dict['device'])
    return inference_results

def bootstrap_convolution_infer(values_for_training_dict:dict) -> dict:
    dataset = values_for_training_dict['dataset']
    model = values_for_training_dict['model']
    name_to_set = {
        'train': {'data': dataset.train_set, 'labels': dataset.train_labels},
        'validation': {'data': dataset.validation_set, 'labels': dataset.validation_labels},
        'test': {'data': dataset.test_set, 'labels': dataset.test_labels}}
    inference_results = {}
    for name, my_set in name_to_set.items():
        if len(my_set['data']) == 0:
            continue
        inference_results[name] = {
            'predictions': model.predict_with_convolution(
                processed_sequences=CNNModel.process_sequences_convolution(
                    sequences=my_set['data'],
                    device=values_for_training_dict['device'],
                )
            ),
            'labels': my_set['labels'],
            'sequences': my_set['data'],
        }
    return inference_results

inference_to_bootstrapper = {
        'LM': bootstrap_LM_infer,
        'convolution': bootstrap_convolution_infer,
    }

def infer_from_parameters(name: str,values_for_training_dict:dict) -> dict:
    supported_training_functions = list(inference_to_bootstrapper.keys())
    if name not in inference_to_bootstrapper.keys():
        raise Exception(f'train_function: {name} not supported. supported training functions: {supported_training_functions}')
    return inference_to_bootstrapper[name](values_for_training_dict)