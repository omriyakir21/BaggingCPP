
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F



class CNNModel(nn.Module):
    AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWY'
    AA_TO_INDEX = {aa: idx for idx, aa in enumerate(AMINO_ACIDS)}
    MAX_LENGTH = 50  # Maximum length of the sequences

    def __init__(self, filters=32, kernel_size=3, num_layers=1,padding = 'valid',dilation = 1):
        super(CNNModel, self).__init__()
        valid_model = CNNModel.check_if_parameters_can_create_valid_model(kernel_size, num_layers, dilation)
        if not valid_model:
            raise ValueError(f"Invalid model parameters: kernel_size={kernel_size}, num_layers={num_layers}, dilation={dilation}.")
        layers = []
        length = 50
        in_channels = 20
        stride = 1
        for _ in range(num_layers):
            layers.append(nn.Conv1d(in_channels=in_channels, out_channels=filters, kernel_size=kernel_size,padding = padding,dilation = dilation))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool1d(kernel_size=2))
            in_channels = filters
            if  padding == 'valid':
                 length = ((length - dilation * (kernel_size - 1) - 1) // stride) + 1  # (length−dilation×(kernel_size−1)−1)//stride+1
                # length = (length - kernel_size + 1)\
            length = length // 2
        self.conv_layers = nn.Sequential(*layers)
        self.fc1 = nn.Linear(filters*length, 128)
        self.fc2 = nn.Linear(128, 1)
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x
    
    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
    
    def predict_with_convolution(self,processed_sequences:torch.Tensor)-> np.ndarray:
        with torch.no_grad():
            self.eval()
            predictions = self(processed_sequences).squeeze(dim=1).cpu().numpy()
        return predictions
    
    @staticmethod
    def check_if_parameters_can_create_valid_model(kernel_size, num_layers, dilation):
        """Check if the parameters can create a valid model."""
        length = 50
        stride = 1
        for _ in range(num_layers):
            length = ((length - dilation * (kernel_size - 1) - 1) // stride) + 1
            length = length // 2
            if length <= 0:
                return False
        return True
    
    @staticmethod
    def one_hot_encode(sequence:list)-> np.ndarray:
        """One-hot encode a sequence of amino acids."""
        encoding = np.zeros((len(sequence), len(CNNModel.AMINO_ACIDS)))
        for i, aa in enumerate(sequence):
            if aa in CNNModel.AA_TO_INDEX:
                encoding[i, CNNModel.AA_TO_INDEX[aa]] = 1
        return encoding

    @staticmethod
    def pad_sequence(encoded_sequence:np.ndarray)-> np.ndarray:
        """Pad the one-hot encoded sequence with zeros to a fixed length."""
        if len(encoded_sequence) >= CNNModel.MAX_LENGTH:
            return encoded_sequence[:CNNModel.MAX_LENGTH]
        else:
            padding = np.zeros((CNNModel.MAX_LENGTH - len(encoded_sequence), len(CNNModel.AMINO_ACIDS)))
            return np.vstack((encoded_sequence, padding))
    
    @staticmethod  
    def process_sequences(sequences: list)-> np.ndarray:
        """Process a list of amino acid sequences by one-hot encoding and padding."""
        processed_sequences = []
        for seq in sequences:
            encoded = CNNModel.one_hot_encode(seq)
            padded = CNNModel.pad_sequence(encoded)
            processed_sequences.append(padded)
        return np.array(processed_sequences)
    
    @staticmethod
    def process_sequences_convolution(sequences:list,device:torch.device)-> torch.Tensor:
        """Process a list of amino acid sequences by one-hot encoding and padding, and convert to PyTorch tensor."""
        dataset = torch.tensor(CNNModel.process_sequences(sequences), dtype=torch.float32).transpose(1, 2).to(device)
        return dataset
    