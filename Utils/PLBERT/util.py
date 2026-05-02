import os
import yaml
import torch
from transformers import AlbertConfig, AlbertModel

# v0.4: per-token language conditioning. Lang ids are added at the embedding
# layer so every downstream consumer (encoder stack, bert_encoder, predictor
# text_encoder, duration_proj, F0/N predictor) sees the language signal.
# 0=mr, 1=en. Optional (None disables — gives v0.1/0.2/0.3 baseline path).
NUM_LANGUAGES = 2


class CustomAlbert(AlbertModel):
    def __init__(self, config):
        super().__init__(config)
        self.lang_embedding = torch.nn.Embedding(NUM_LANGUAGES, config.embedding_size)
        torch.nn.init.normal_(self.lang_embedding.weight, std=0.02)

    def forward(self, input_ids=None, lang_ids=None, attention_mask=None, **kwargs):
        if lang_ids is not None and input_ids is not None:
            word_embeds = self.embeddings.word_embeddings(input_ids)
            lang_embeds = self.lang_embedding(lang_ids)
            outputs = super().forward(
                inputs_embeds=word_embeds + lang_embeds,
                attention_mask=attention_mask,
                **kwargs,
            )
        else:
            outputs = super().forward(
                input_ids=input_ids, attention_mask=attention_mask, **kwargs
            )
        return outputs.last_hidden_state


def load_plbert(log_dir):
    config_path = os.path.join(log_dir, "config.yml")
    plbert_config = yaml.safe_load(open(config_path))

    albert_base_configuration = AlbertConfig(**plbert_config['model_params'])
    bert = CustomAlbert(albert_base_configuration)

    files = os.listdir(log_dir)
    ckpts = []
    for f in os.listdir(log_dir):
        if f.startswith("step_"): ckpts.append(f)

    iters = [int(f.split('_')[-1].split('.')[0]) for f in ckpts if os.path.isfile(os.path.join(log_dir, f))]
    iters = sorted(iters)[-1]

    checkpoint = torch.load(log_dir + "/step_" + str(iters) + ".t7", map_location='cpu')
    state_dict = checkpoint['net']
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] # remove `module.`
        if name.startswith('encoder.'):
            name = name[8:] # remove `encoder.`
            new_state_dict[name] = v
    del new_state_dict["embeddings.position_ids"]
    bert.load_state_dict(new_state_dict, strict=False)
    
    return bert
