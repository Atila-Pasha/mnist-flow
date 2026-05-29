import torch.nn as nn


class MLP(nn.Module):

    def __init__(
        self,
        input_size,
        hidden_1,
        hidden_2,
        num_classes
    ):

        super().__init__()

        self.layers = nn.Sequential(

            nn.Linear(
                input_size,
                hidden_1
            ),

            nn.ReLU(),

            nn.Linear(
                hidden_1,
                hidden_2
            ),

            nn.ReLU(),

            nn.Linear(
                hidden_2,
                num_classes
            )
        )

    def forward(self, x):

        x = x.view(
            x.size(0),
            -1
        )

        return self.layers(x)