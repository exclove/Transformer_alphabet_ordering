import argparse

def get_config():
    parser = argparse.ArgumentParser(description="Transformer Model Hyperparameters")
    parser.add_argument("--embed_size", type=int, default=256, help="Embedding size for the model")
    parser.add_argument("--num_layers", type=int, default=6, help="Number of Transformer layers")
    parser.add_argument("--forward_expansion", type=int, default=4, help="Expansion factor for feedforward layers")
    parser.add_argument("--heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--dropout", type=float, default=0, help="Dropout rate")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--max_length", type=int, default=100, help="Maximum sequence length")

    return parser

if __name__ == "__main__":
    parser = get_config()
    args = parser.parse_args()
    print(args.embed_size)