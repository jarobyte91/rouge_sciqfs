import torch
import torch.nn as nn
import torch.utils.data as tud
import numpy as np
import pandas as pd

class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def fit(
        self,
        X, 
        Y,
        batch_size = 8,
        epochs = 10,
        learning_rate = 10 ** -4,
        weight_decay = 0
        
    ):
        dataset = tud.TensorDataset(X, Y)
        loader = tud.DataLoader(
            dataset,
            batch_size = batch_size,
            shuffle = True
        )
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(
            self.parameters(), 
            lr = learning_rate,
            weight_decay = weight_decay
        )
        train_log = []
        for i in range(1, epochs + 1):
            losses = []
            for x, y in (iter(loader)):
                preds = self.forward(x).squeeze()
                loss = criterion(preds, y.float())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
            epoch_loss = sum(losses) / len(losses)
            train_log.append(epoch_loss)
            print(f"  epoch {i}, loss {epoch_loss:.6f}")
        results = pd.Series(range(1, epochs + 1), name = "epoch").to_frame()\
        .assign(
            total_epochs = epochs,
            train_loss = train_log
        )
        return results

    def predict(self, X, batch_size = 16):
        test_dataset = tud.TensorDataset(X)
        test_loader = tud.DataLoader(test_dataset, batch_size = batch_size)
        predictions = []
        for (b,) in test_loader:
            p = self.forward(b.cuda()).cpu().detach().numpy()#.squeeze()
            predictions.append(p)
        predictions = np.concatenate(predictions).squeeze()
        return predictions

class MLP(Model):
    def __init__(
        self,
        input_size = 768,
        hidden_size = 100,
        num_layers = 2
    ):
        super().__init__()
        self.input_layer = nn.Linear(input_size, hidden_size)
        self.layers = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for i in range(num_layers)])
        self.output_layer = nn.Linear(hidden_size, 1)
        print("MLP Classfier")
        print(f"Parameters: {sum([x.numel() for x in self.parameters()]):,}")
        
    def forward(self, X):
        X = self.input_layer(X)
        for l in self.layers:
            X = l(X)
        return self.output_layer(X).sigmoid()      
        
class LSTM(Model):
    def __init__(
        self,
        input_size = 768,
        hidden_size = 100,
        num_layers = 2
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size = input_size,
            hidden_size = hidden_size,
            num_layers = num_layers,
            bidirectional = True
        )
        self.out_layer = nn.Linear(2 * hidden_size, 1)
        print("LSTM Classfier")
        print(f"Parameters: {sum([x.numel() for x in self.parameters()]):,}")
        
    def forward(self, x):
        x, (last_hidden, last_memory) = self.lstm(x)
        return self.out_layer(x).sigmoid()
    
class Transformer(Model):
    def __init__(
        self,
        dim_feedforward = 768,
        attention_heads = 4,
        num_layers = 4,
        d_model = 768,
        max_sequence_length = 512
    ):
        super().__init__()
        self.positional_embeddings = nn.Embedding(
            max_sequence_length, 
            d_model
        )        
        self.transformer_layer = nn.TransformerEncoderLayer(
            d_model = d_model,
            dim_feedforward = dim_feedforward,
            nhead = attention_heads
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer = self.transformer_layer,
            num_layers = num_layers
        )
        self.out_layer = nn.Linear(d_model, 1)
        print("Transformer Classfier")
        print(f"Parameters: {sum([x.numel() for x in self.parameters()]):,}")
        
    def forward(self, x):
        x_positional = torch.arange(
            x.shape[1], 
            device = next(self.parameters()).device
        )\
        .repeat((x.shape[0], 1))
        x_positional = self.positional_embeddings(x_positional)
        x = (x + x_positional).transpose(0, 1)
        x = self.encoder.forward(src = x).transpose(0, 1)
        return self.out_layer(x).sigmoid()
