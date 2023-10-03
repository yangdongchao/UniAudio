import torch
import logging
import json

# For some data types that are large and can hardly be fully stored in memory,
# We do offline tokenization and save them as codec sequences. e.g., audio
def load_pt_data(f):
    return torch.load(f, map_location='cpu')

def load_text_data(f):
    lines = open(f, encoding='utf-8').readlines()
    lines = [line.strip().split() for line in lines]
    ret = {}
    for line in lines:
        if len(line) < 2:
            logging.warning(f"find an empty entry: {line}")
            continue
        example_id, ctx = line[0], " ".join(line[1:])
        ret[example_id] = ctx
    return ret

def unified_loading(f):
    """ allow both format """
    if f.endswith('.pt'):
        return load_pt_data(f)
    else:
        return load_text_data(f)

loading_methods = {
    'audio': load_pt_data,
    'audio_prompt': unified_loading,
    'text': load_pt_data,
    'text_emb': load_pt_data,
    'phone': load_pt_data,
    'semantic': load_pt_data,
    'class_event': load_pt_data,
    'text_t5': unified_loading,
    'sv_bool': unified_loading,
    'sing_phone': unified_loading,
    'sing_midi': load_pt_data,
}
        
# 2. This part defines all valid task format.
# The data format of each task is defined as below:
# (1)   keys: data keys in order. This determines the order of the components in the sequences
# (2)   type: type of each data key. It determines the tokenizer for each data key
# (3)   features: some features belong to the examples but are not in the training sequence. e.g., speaker-id
# (4)   loss_key: key to predict. it determines which data key the loss should be computed on.
# (5)   encoder_keys: keys that are placed in encoder input when using the encoder-decoder format. 
#         Should always be the first several entry in "keys"
#         If this is set to None or [], it means encoder-decoder format is not supported, e.g., LM
# Note: you may need to add more type beside the current ones. However, to support a new type, you should
# provide a new tokenizer inherited from the AbsTokenizer
# Maybe some TODO: (1) features are text-only -> maybe more types
#                  (2) only one loss_key -> maybe support more than one

tts_format = {
    'keys': ["phone_seq", "prompt_seq", "audio_seq"],
    'type': ["phone", "audio_prompt", "audio"],
    'features': [],
    'loss_key': 'audio_seq',
    'encoder_keys': ['phone_seq', 'prompt_seq'],
}
plain_tts_format = {
    'keys': ["phone_seq", "audio_seq"],
    'type': ["phone", "audio"],
    'features': [],
    'loss_key': 'audio_seq',
    'encoder_keys': ['phone_seq'],
}
lm_format = {
    'keys': ["text_seq"],
    'type': ["text"],
    'features': [],
    'loss_key': 'text_seq',
    'encoder_keys': None,
}
asr_format = {
    'keys': ['semantic_seq', 'text_seq'],
    'type': ['semantic', 'text'],
    'features': [],
    'loss_key': 'text_seq',
    'encoder_keys': ['semantic_seq'],
}
phone_to_semantic_format = {
    'keys': ["phone_seq", "semantic_seq"],
    'type': ["phone", "semantic"],
    'features': [],
    'loss_key': 'semantic_seq',
    'encoder_keys': ['phone_seq'],
}
semantic_to_acoustic_format = {
    'keys': ["semantic_seq", "audio_seq"],
    'type': ["semantic", "audio"],
    'features': [],
    'loss_key': 'audio_seq',
    'encoder_keys': ['semantic_seq']
}

t2a_format = {
    'keys': ["text_emb_seq", "audio_seq"],
    'type': ["text_emb", "audio"],
    'features': [],
    'loss_key': 'audio_seq',
}
SE_format = {
    'keys': ["noise_seq", "audio_seq"],
    'type': ["audio", "audio"],
    'features': [],
    'loss_key': 'audio_seq',
}
VC_format = {
    'keys': ["semantic_seq", "prompt_seq", "audio_seq"],
    'type': ["semantic",  "audio_prompt", "audio"],
    'features': [],
    'loss_key': 'audio_seq',
}
AT_format = {
    'keys': ["audio_seq", "class_seq"],
    'type': ["audio",  "class_event"],
    'features': [],
    'loss_key': 'class_seq',
}
Spex_format = {
    'keys': ["noise_seq", "prompt_seq", "audio_seq"],
    'type': ["audio", "audio_prompt", "audio"],
    'features': [],
    'loss_key': 'audio_seq',
}  # we fix the audio to audio_prompt in the second version
TTA_format = {
    'keys': ["rvq_seq", "audio_seq"],
    'type': ["rvq", "audio"],
    'features': [],
    'loss_key': 'audio_seq',
}
TSS_format = {
    'keys': ["text_t5_seq", "audio_seq"],
    'type': ["text_t5", "audio"],
    'features': [],
    'loss_key': 'audio_seq',
}
SV_format = {
    'keys': ['audio_seq', 'prompt_seq', 'label'],
    'type': ['audio', 'audio_prompt', 'sv_bool'],
    'features': [],
    'loss_key': 'label',
}
Sing_format = {
    'keys': ["sing_phone_seq", "sing_midi_seq", "prompt_seq", "audio_seq"],
    'type': ["sing_phone", "sing_midi", "audio_prompt", "audio"],
    'features': [],
    'loss_key': 'audio_seq',
}
TTM_format = {
    'keys': ["text_t5_seq", "audio_seq"],
    'type': ["text_t5", "audio"],
    'features': [],
    'loss_key': 'audio_seq',
}
AudioEdit_format = {
    'keys': ["text_t5_seq", "audio_source_seq", "audio_seq"],
    'type': ["text_t5", "audio", "audio"],
    'features': [],
    'loss_key': 'audio_seq',
}
InstructTTS_format = {
    'keys': ["phone_seq", "text_t5_seq", "audio_seq"],
    'type': ["phone", "text_t5", "audio"],
    'features': [],
    'loss_key': 'audio_seq',
}
RIR_format = {
    'keys': ["noise_seq", "audio_seq"],
    'type': ["audio", "audio"],
    'features': [],
    'loss_key': 'audio_seq',
}
speech_edit_format = {
    'keys': ["phone_seq", "corrupted_audio_seq", "audio_seq"],
    'type': ["phone", "audio", "audio"],
    'features': [],
    'loss_key': 'audio_seq',
    'encoder_keys': ['phone_seq', 'audio_seq'],
}
task_formats = {
    'lm': lm_format,
    'tts': tts_format,
    'asr': asr_format,
    'plain_tts': plain_tts_format,
    'phone_to_semantic': phone_to_semantic_format,
    'semantic_to_acoustic': semantic_to_acoustic_format,
    't2a': t2a_format,
    'SE': SE_format,
    'VC': VC_format,
    'AT': AT_format,
    'Spex': Spex_format,
    'TTA': TTA_format,
    'TSS': TSS_format,
    'SV': SV_format,
    'sing': Sing_format,
    'TTM': TTM_format,
    'Audit': AudioEdit_format,
    'InstructTTS': InstructTTS_format,
    'Speech_RIR': RIR_format,
    'speech_edit': speech_edit_format,
}

# 3. This part defins how data is loaded in the data_dict at the loading stage
# It load all data into the memory according to the task format definition.
# It roughly compute the length of each data key, along with the length of the
# whole sequence that can be used for batchfy.
# However, it doesn't do any tokenization and data combination: they are done
# in the collate_fn
# Note, since all data is fully stored in the memory during training, the data
# should only be in light format: e.g., Text / Codec.
# Other raw data are not supported since volume is large: e.g., raw audio
# / image / SSL model embeddings (they are computed on-the-fly in the tokenizers).
def load_data_for_all_tasks(json_files):
    """ accept and parse multiple json_files, each of which represents a task dataset"""
    data_dict = {}
    for json_file in json_files:
        dataset_json = json.load(open(json_file)) 
        logging.info(f"loading dataset file: {json_file} for {dataset_json['task']} task") 
        print(f"loading dataset file: {json_file} for {dataset_json['task']} task")                  
        task_data = load_data_for_one_task(dataset_json)
        data_dict.update(task_data)
    logging.info(f"from all json files, we have {len(data_dict)} examples")
    print(f"from all json files, we have {len(data_dict)} examples")
    return data_dict

def load_data_for_one_task(dataset_json):
    task_type = dataset_json['task']
    task_format = task_formats[task_type]

    # load data for each data key
    data_dict = {}
    for key, data_type in zip(task_format['keys'], task_format['type']):
        if key not in dataset_json['keys']:
            raise ValueError(f"For task {task_type}, data key {key} is needed but missing.")

        logging.info(f"loading file: {dataset_json['keys'][key]} as key: {key}")
        print(f"loading file: {dataset_json['keys'][key]} as key: {key}")
        this_data_dict = loading_methods[data_type](dataset_json['keys'][key])
        this_data_dict = {f"{dataset_json['task']}_{k}": v 
                for k, v in this_data_dict.items()
        }
        for example_id, data in this_data_dict.items():
            if example_id not in data_dict:
                data_dict[example_id] = {}
            data_dict[example_id][key] = data

    # load data for each feature
    for feat in task_format['features']:
        if feat not in dataset_json['features']:
            raise ValueError(f"For task {task_type}, data feature {feat} is needed but missing")

        feature_file = dataset_json['features'][feat]
        logging.info(f"loading file: {feature_file} as a feature: {feat}")

        feature_dict = open(feature_file).readlines()
        feature_dict = [line.strip().split() for line in feature_dict]
        feature_dict = {line[0]: line[1:] for line in feature_dict}

        for example_id, data in feature_dict.items():
            if example_id not in data_dict:
                data_dict[example_id] = {}
            data_dict[example_id][feat] = data

    # Validate the data: remove the examples when some entries are missing.
    # add the task label after validation
    example_ids = list(data_dict.keys())
    for example_id in example_ids:
        for key in task_format['keys'] + task_format['features']:
            if key not in data_dict[example_id]:
                del data_dict[example_id]
                logging.warning(f"{task_type} example {example_id} is removed since {key} is missing")
                #print(f"{task_type} example {example_id} is removed since {key} is missing")
                break

    example_ids = list(data_dict.keys())
    for example_id in example_ids:
        data_dict[example_id]['task'] = task_type
        data_dict[example_id]['loss_key'] = task_format['loss_key']

    logging.info(f"done loading this raw data dict: {len(data_dict)} valid examples")
    print(f"done loading this raw data dict: {len(data_dict)} valid examples")

    return data_dict

if __name__ == "__main__":
    pass

