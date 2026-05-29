import yaml

import torch

from model import MLP


with open(
    "configs/config.yaml",
    "r"
) as file:

    config = yaml.safe_load(file)


device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


model = MLP(
    config["input_size"],
    config["hidden_1"],
    config["hidden_2"],
    config["num_classes"]
).to(device)


model.load_state_dict(
    torch.load(
        "checkpoints/mnist_mlp.pth",
        map_location=device
    )
)

model.eval()

print("Model loaded successfully.")