import torch
from torch import nn


def _init_gru_parameters(gru_module):
    """Apply orthogonal/xavier initialisation to GRU parameters."""
    for name, param in gru_module.named_parameters():
        if "weight_hh" in name:
            nn.init.orthogonal_(param)
        if "weight_ih" in name:
            nn.init.xavier_uniform_(param)

class GRUDecoder(nn.Module):
    '''
    Defines the GRU decoder

    This class combines day-specific input layers, a GRU, and an output classification layer
    '''
    def __init__(self,
                 neural_dim,
                 n_units,
                 n_days,
                 n_classes,
                 rnn_dropout = 0.0,
                 input_dropout = 0.0,
                 n_layers = 5, 
                 patch_size = 0,
                 patch_stride = 0,
                 ):
        '''
        neural_dim  (int)      - number of channels in a single timestep (e.g. 512)
        n_units     (int)      - number of hidden units in each recurrent layer - equal to the size of the hidden state
        n_days      (int)      - number of days in the dataset
        n_classes   (int)      - number of classes 
        rnn_dropout    (float) - percentage of units to droupout during training
        input_dropout (float)  - percentage of input units to dropout during training
        n_layers    (int)      - number of recurrent layers 
        patch_size  (int)      - the number of timesteps to concat on initial input layer - a value of 0 will disable this "input concat" step 
        patch_stride(int)      - the number of timesteps to stride over when concatenating initial input 
        '''
        super(GRUDecoder, self).__init__()
        
        self.neural_dim = neural_dim
        self.n_units = n_units
        self.n_classes = n_classes
        self.n_layers = n_layers 
        self.n_days = n_days

        self.rnn_dropout = rnn_dropout
        self.input_dropout = input_dropout
        
        self.patch_size = patch_size
        self.patch_stride = patch_stride

        # Parameters for the day-specific input layers
        self.day_layer_activation = nn.Softsign() # basically a shallower tanh 

        # Set weights for day layers to be identity matrices so the model can learn its own day-specific transformations
        self.day_weights = nn.ParameterList(
            [nn.Parameter(torch.eye(self.neural_dim)) for _ in range(self.n_days)]
        )
        self.day_biases = nn.ParameterList(
            [nn.Parameter(torch.zeros(1, self.neural_dim)) for _ in range(self.n_days)]
        )

        self.day_layer_dropout = nn.Dropout(input_dropout)
        
        self.input_size = self.neural_dim

        # If we are using "strided inputs", then the input size of the first recurrent layer will actually be in_size * patch_size
        if self.patch_size > 0:
            self.input_size *= self.patch_size

        self.gru = nn.GRU(
            input_size = self.input_size,
            hidden_size = self.n_units,
            num_layers = self.n_layers,
            dropout = self.rnn_dropout,
            batch_first = True, # The first dim of our input is the batch dim
            bidirectional = False,
        )

        # Set recurrent units to have orthogonal param init and input layers to have xavier init
        _init_gru_parameters(self.gru)

        # Prediciton head. Weight init to xavier
        self.out = nn.Linear(self.n_units, self.n_classes)
        nn.init.xavier_uniform_(self.out.weight)

        # Learnable initial hidden states
        self.h0 = nn.Parameter(nn.init.xavier_uniform_(torch.zeros(1, 1, self.n_units)))

    def forward(self, x, day_idx, states = None, return_state = False):
        '''
        x        (tensor)  - batch of examples (trials) of shape: (batch_size, time_series_length, neural_dim)
        day_idx  (tensor)  - tensor which is a list of day indexs corresponding to the day of each example in the batch x. 
        '''

        # Apply day-specific layer to (hopefully) project neural data from the different days to the same latent space
        day_weights = torch.stack([self.day_weights[i] for i in day_idx], dim=0)
        day_biases = torch.cat([self.day_biases[i] for i in day_idx], dim=0).unsqueeze(1)

        x = torch.einsum("btd,bdk->btk", x, day_weights) + day_biases
        x = self.day_layer_activation(x)

        # Apply dropout to the ouput of the day specific layer
        if self.input_dropout > 0:
            x = self.day_layer_dropout(x)

        # (Optionally) Perform input concat operation
        if self.patch_size > 0: 
  
            x = x.unsqueeze(1)                      # [batches, 1, timesteps, feature_dim]
            x = x.permute(0, 3, 1, 2)               # [batches, feature_dim, 1, timesteps]
            
            # Extract patches using unfold (sliding window)
            x_unfold = x.unfold(3, self.patch_size, self.patch_stride)  # [batches, feature_dim, 1, num_patches, patch_size]
            
            # Remove dummy height dimension and rearrange dimensions
            x_unfold = x_unfold.squeeze(2)           # [batches, feature_dum, num_patches, patch_size]
            x_unfold = x_unfold.permute(0, 2, 3, 1)  # [batches, num_patches, patch_size, feature_dim]

            # Flatten last two dimensions (patch_size and features)
            x = x_unfold.reshape(x.size(0), x_unfold.size(1), -1) 
        
        # Determine initial hidden states
        if states is None:
            states = self.h0.expand(self.n_layers, x.shape[0], self.n_units).contiguous()

        # Pass input through RNN 
        output, hidden_states = self.gru(x, states)

        # Compute logits
        logits = self.out(output)
        
        if return_state:
            return logits, hidden_states
        
        return logits


class DCoNDDecoder(nn.Module):
    """GRU-based decoder that predicts diphone distributions with marginalisation."""

    def __init__(
        self,
        neural_dim,
        n_units,
        n_days,
        n_classes,
        rnn_dropout = 0.0,
        input_dropout = 0.0,
        n_layers = 5,
        patch_size = 0,
        patch_stride = 0,
        sil_phoneme_id = None,
    ):
        super().__init__()

        if n_classes <= 1:
            raise ValueError('n_classes must include the CTC blank and at least one phoneme.')

        if sil_phoneme_id is None:
            sil_phoneme_id = n_classes - 1

        if sil_phoneme_id <= 0 or sil_phoneme_id >= n_classes:
            raise ValueError('sil_phoneme_id must reference a non-blank phoneme class.')

        self.neural_dim = neural_dim
        self.n_units = n_units
        self.n_layers = n_layers
        self.n_days = n_days
        self.n_classes = n_classes
        self.rnn_dropout = rnn_dropout
        self.input_dropout = input_dropout
        self.patch_size = patch_size
        self.patch_stride = patch_stride

        self.n_phonemes = n_classes - 1  # exclude CTC blank
        self.sil_zero_index = sil_phoneme_id - 1

        if self.sil_zero_index < 0 or self.sil_zero_index >= self.n_phonemes:
            raise ValueError('sil_phoneme_id is out of bounds for the provided class mapping.')

        # Day specific parameters
        self.day_layer_activation = nn.Softsign()
        self.day_weights = nn.ParameterList(
            [nn.Parameter(torch.eye(self.neural_dim)) for _ in range(self.n_days)]
        )
        self.day_biases = nn.ParameterList(
            [nn.Parameter(torch.zeros(1, self.neural_dim)) for _ in range(self.n_days)]
        )
        self.day_layer_dropout = nn.Dropout(input_dropout)

        self.input_size = self.neural_dim
        if self.patch_size > 0:
            self.input_size *= self.patch_size

        self.gru = nn.GRU(
            input_size=self.input_size,
            hidden_size=self.n_units,
            num_layers=self.n_layers,
            dropout=self.rnn_dropout,
            batch_first=True,
            bidirectional=False,
        )
        _init_gru_parameters(self.gru)

        # Output heads
        self.diphone_head = nn.Linear(self.n_units, self.n_phonemes * self.n_phonemes)
        nn.init.xavier_uniform_(self.diphone_head.weight)

        self.blank_head = nn.Linear(self.n_units, 1)
        nn.init.xavier_uniform_(self.blank_head.weight)

        self.h0 = nn.Parameter(nn.init.xavier_uniform_(torch.zeros(1, 1, self.n_units)))

    def forward(self, x, day_idx, states=None, return_state=False):
        day_weights = torch.stack([self.day_weights[i] for i in day_idx], dim=0)
        day_biases = torch.cat([self.day_biases[i] for i in day_idx], dim=0).unsqueeze(1)

        x = torch.einsum("btd,bdk->btk", x, day_weights) + day_biases
        x = self.day_layer_activation(x)

        if self.input_dropout > 0:
            x = self.day_layer_dropout(x)

        if self.patch_size > 0:
            x = x.unsqueeze(1)
            x = x.permute(0, 3, 1, 2)
            x_unfold = x.unfold(3, self.patch_size, self.patch_stride)
            x_unfold = x_unfold.squeeze(2)
            x_unfold = x_unfold.permute(0, 2, 3, 1)
            x = x_unfold.reshape(x.size(0), x_unfold.size(1), -1)

        if states is None:
            states = self.h0.expand(self.n_layers, x.shape[0], self.n_units).contiguous()

        output, hidden_states = self.gru(x, states)

        diphone_logits = self.diphone_head(output)
        blank_logits = self.blank_head(output)

        combined_logits = torch.cat([blank_logits, diphone_logits], dim=-1)
        combined_log_probs = combined_logits.log_softmax(dim=-1)

        blank_log_probs = combined_log_probs[..., :1]
        diphone_log_probs = combined_log_probs[..., 1:]
        diphone_log_probs = diphone_log_probs.view(
            combined_log_probs.size(0),
            combined_log_probs.size(1),
            self.n_phonemes,
            self.n_phonemes,
        )

        phoneme_log_probs = torch.logsumexp(diphone_log_probs, dim=-2)
        phoneme_logits = torch.logsumexp(
            diphone_logits.view(diphone_logits.size(0), diphone_logits.size(1), self.n_phonemes, self.n_phonemes),
            dim=-2,
        )
        phoneme_logits = torch.cat([blank_logits, phoneme_logits], dim=-1)

        outputs = {
            'phoneme_log_probs': torch.cat([blank_log_probs, phoneme_log_probs], dim=-1),
            'diphone_log_probs': combined_log_probs,
            'phoneme_logits': phoneme_logits,
            'blank_log_probs': blank_log_probs,
        }

        if return_state:
            return outputs, hidden_states

        return outputs
