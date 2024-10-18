import torch
from dataset import get_dataset
from models import get_model
from train import train_model
import yaml

def main():
    with open("config.yaml", "r") as file:
        config = yaml.safe_load(file)

    dataset_name = config['dataset_name']
    model_name = config['model_name']
    num_epochs = config['num_epochs']
    batch_size = config['batch_size']
    learning_rate = config['learning_rate']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_loader, test_loader = get_dataset(dataset_name, batch_size)
    model = get_model(model_name, num_classes=10)

    train_model(model, train_loader, test_loader, num_epochs, learning_rate, device)

if __name__ == "__main__":
    main()
