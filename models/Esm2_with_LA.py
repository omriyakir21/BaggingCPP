import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss, BCEWithLogitsLoss
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers import EsmPreTrainedModel, EsmModel,PretrainedConfig, AutoModel
import os
class LightAttention(nn.Module):
    def __init__(self, model_config,device):
        super(LightAttention, self).__init__()
        self.use_max = model_config.use_max
        # print(f"in_channels: {model_config.hidden_size}, out_channels: {model_config.dout}, kernel_size: {model_config.kernel_size}")
        self.conv_e = nn.Conv1d(
            in_channels=model_config.hidden_size, 
            out_channels=model_config.dout // 2 if model_config.use_max else model_config.dout, 
            kernel_size=model_config.kernel_size, 
            padding=model_config.kernel_size // 2,
            bias=True,
        )
        self.conv_v = nn.Conv1d(
            in_channels=model_config.hidden_size, 
            out_channels=model_config.dout // 2 if model_config.use_max else model_config.dout, 
            kernel_size=model_config.kernel_size, 
            padding=model_config.kernel_size // 2,
            bias=True,
        )

    def forward(self, x):
        # x has shape (batch, sequence_length, hidden_size)
        # Transpose x to (batch, hidden_size, sequence_length) for 1D conv
        x = x.transpose(1, 2)
        e = self.conv_e(x)  # shape: (batch, dout, sequence_length)
        v = self.conv_v(x)  # shape: (batch, dout, sequence_length)
        # Apply softmax along the sequence length dimension to get attention weights
        a = F.softmax(e, dim=2)
        att = a * v
        # Compute the weighted sum over the sequence dimension
        sum_att = torch.sum(att, dim=2)  # shape: (batch, dout)
        if self.use_max:
            # compute the max over the sequence dimension
            max_att, _ = torch.max(v, dim=2)
            #concat the sum and max
            final_att = torch.cat((sum_att, max_att), dim=1)
            return final_att
        return sum_att


class ESMWithLightAttentionHead(nn.Module):
    def __init__(self, config,device,loss_fct):
        # Initialize using the parent class (this sets up self.esm, among other things)
        super().__init__()
        self.config = config
        # self.esm = EsmModel.from_pretrained(config._name_or_path,config=config, add_pooling_layer=False)
        self.esm = AutoModel.from_pretrained(config._name_or_path,config=config, add_pooling_layer=False,trust_remote_code=True)
        self.num_labels = config.num_labels
        self.loss_fct = loss_fct
        self.light_attention = LightAttention(model_config=config,device=device)
        self.dropout = nn.Dropout(config.hidden_dropout_prob) if  hasattr(config, "hidden_dropout_prob") else nn.Identity()
        self.classifier = nn.Linear(config.dout, config.num_labels)

        self.init_weights()
        for name,param in self.named_parameters():
            if torch.isnan(param).any():
                raise ValueError(f"NaN detected in model weights for {name}")
    
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        head_mask=None,
        labels = None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        return_embeddings: bool = False,
    ):
        

        # Call self.esm to process the inputs
        outputs = self.esm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # Get the sequence output
        sequence_output = outputs[0]
        # Use dropout then aggregate via LightAttention
        x = self.dropout(sequence_output)
        x = self.light_attention(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        embeddings = x
        logits = self.classifier(x).squeeze(dim=-1)
        if labels is not None:
            labels = labels.to(logits.device)
            if self.num_labels == 1:
                loss = self.loss_fct(logits, labels.float().squeeze())
            else:
                loss = self.loss_fct(logits, labels)
        else:
            loss = None

        output = SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

        if return_embeddings:
            # Return the embeddings along with the output
            return output, embeddings
        else:
            # Return only the output
            return output
    
    


    
    def init_weights(self):
        # Initialize weights for the model
        nn.init.xavier_uniform_(self.light_attention.conv_e.weight)
        nn.init.xavier_uniform_(self.light_attention.conv_v.weight)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)
            # if "conv" in name:
            #     nn.init.uniform_(param, a=0, b=0.1)
            # else:
            #     nn.init.xavier_uniform_(param)
    
    def save_LA_head(self, save_dir: str) -> None:
        os.makedirs(save_dir, exist_ok=True)
        layers = {
            "light_attention": self.light_attention.state_dict(),
            "dropout": self.dropout.state_dict(),
            "classifier": self.classifier.state_dict()
        }
        layers_file = os.path.join(save_dir, "LA_head_layers.pth")
        torch.save(layers, layers_file)
        print(f"Selected layers saved to {layers_file}")