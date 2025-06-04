import torch
import torch.nn as nn
import torch.nn.functional as F
class Encoder(nn.Module):
    def __init__(self, input_size=1, hidden_size=30, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True)

    def forward(self, x):
        outputs, (hidden, cell) = self.lstm(x)
        return hidden, cell

class Decoder(nn.Module):
    def __init__(self, input_size=1, hidden_size=30, output_size=30):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size + hidden_size,
                            hidden_size=hidden_size,
                            num_layers=1,
                            batch_first=True)
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, input, hidden, cell, encoder_outputs):
        seq_len = encoder_outputs.shape[1]
        hidden_repeated = hidden.repeat(seq_len, 1, 1).permute(1, 0, 2)
        energy = self.attention(torch.cat((hidden_repeated, encoder_outputs), dim=2))
        attention_weights = torch.softmax(energy, dim=1)
        context_vector = torch.sum(attention_weights * encoder_outputs, dim=1).unsqueeze(1)
        lstm_input = torch.cat([input, context_vector], dim=2)
        dec_output, (dec_hidden, dec_cell) = self.lstm(lstm_input, (hidden, cell))
        return dec_output

class Output(nn.Module):
    def __init__(self, input_size = 30, hidden_size = 30, output_size = 1):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size,hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size,output_size),
        )
    def forward(self, dec_output):
        return self.fc(dec_output)

class Seq2Seq(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.output = Output()
    def forward(self, counter_seq):
        batch_size = counter_seq.size(0)
        encoder_outputs, (hidden, cell) = self.encoder.lstm(counter_seq)
        dec_input = torch.ones(batch_size, 1, 1, device=counter_seq.device)
        dec_output = self.decoder(dec_input, hidden, cell, encoder_outputs)
        lin_in = dec_output.squeeze(1)
        prediction = self.output(lin_in)
        return prediction
