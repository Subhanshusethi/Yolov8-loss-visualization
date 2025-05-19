class LossModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.layern = nn.Sequential(
            nn.Linear(4,256),
            nn.ReLU(),
            # nn.Dropout(0.1),
            nn.Linear(256,64),
            nn.ReLU(),
            nn.Linear(64,4)
        )

    def forward(self,x):
        x = self.layern(x)
        return x
