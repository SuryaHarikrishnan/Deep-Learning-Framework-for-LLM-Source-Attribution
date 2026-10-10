import copy
import torch
import torch.nn as nn
from src.evaluate import predict, report
from src.utils import set_seed

def run(model, train_loader, val_loader, test_loader, labels,
        epochs=10, lr=1e-3, fig_path=None, seed=42):
    # shared by the CNN and RNN so both are trained the same way
    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    best_acc, best_state = -1, None

    for ep in range(epochs):
        # train for one pass over the training data
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()                   # clear old gradients
            loss_fn(model(x), y).backward()   # compute error and gradients
            opt.step()                        # update the weights

        # check accuracy on validation data
        yv, pv = predict(model, val_loader, device)
        val_acc = report(yv, pv, labels)["accuracy"]
        print(f"epoch {ep + 1}: val acc {val_acc:.4f}")

        # remember the weights from the best epoch
        if val_acc > best_acc:
            best_acc, best_state = val_acc, copy.deepcopy(model.state_dict())

    # reload the best weights and score once on the test set
    model.load_state_dict(best_state)
    yt, pt = predict(model, test_loader, device)
    return report(yt, pt, labels, fig_path)