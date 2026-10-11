from src.data.dataset import get_loaders
from src.models.cnn import TextCNN
from src.train import run

def main():
    # Load the dataset using the model's generated answers
    train_loader, val_loader, test_loader, vocab, labels = get_loaders(
        mode="output"
    )

    # Initialize the CNN
    model = TextCNN(
        vocab_size=len(vocab),
        num_classes=len(labels)
    )

    print("Starting CNN training...")

    # Train and evaluate using the shared pipeline
    results = run(
        model,
        train_loader,
        val_loader,
        test_loader,
        labels,
        epochs=10,
        lr=0.001
    )

    print("CNN results:")
    print(results)


if __name__ == "__main__":
    main()