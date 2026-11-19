import torch
import torch.nn as nn


class RiskLSTM(nn.Module):

    def __init__(
        self,
        input_size=14,
        hidden_size=64,
        num_layers=2,
        num_classes=2,
        dropout=0.3
    ):
        super().__init__()

        # --------------------------------------------------
        # LSTM
        # --------------------------------------------------

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )


        # --------------------------------------------------
        # FINAL CLASSIFIER
        # --------------------------------------------------

        self.classifier = nn.Linear(
            hidden_size,
            num_classes
        )


    def forward(self, x):

        # x shape:
        # [batch_size, sequence_length, features]
        #
        # Example:
        # [32, 15, 14]

        lstm_output, (hidden_state, cell_state) = self.lstm(x)


        # hidden_state shape:
        #
        # [num_layers, batch_size, hidden_size]
        #
        # Example:
        # [2, 32, 64]


        # Take the hidden state from
        # the FINAL LSTM layer.

        final_hidden = hidden_state[-1]


        # final_hidden:
        # [batch_size, hidden_size]
        #
        # Example:
        # [32, 64]


        logits = self.classifier(
            final_hidden
        )


        # logits:
        # [batch_size, num_classes]
        #
        # Example:
        # [32, 2]

        return logits


# --------------------------------------------------
# TEST MODEL
# --------------------------------------------------

if __name__ == "__main__":

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "Using device:",
        device
    )


    model = RiskLSTM().to(
        device
    )


    # Fake batch matching our actual dataset:
    #
    # 32 sequences
    # 15 time steps
    # 14 features

    x = torch.randn(
        32,
        15,
        14
    ).to(device)


    with torch.no_grad():

        output = model(x)


    print(
        "Input shape:",
        x.shape
    )

    print(
        "Output shape:",
        output.shape
    )


    total_parameters = sum(
        p.numel()
        for p in model.parameters()
    )


    trainable_parameters = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )


    print(
        "Total parameters:",
        total_parameters
    )

    print(
        "Trainable parameters:",
        trainable_parameters
    )