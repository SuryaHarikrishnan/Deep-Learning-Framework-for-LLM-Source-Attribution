import torch
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

@torch.no_grad()  # no gradients needed when only predicting
def predict(model, loader, device):
    # run the model over a dataset, return true labels and predicted labels
    model.eval()
    ys, ps = [], []
    for x, y in loader:
        # the highest score is the predicted LLM
        ps += model(x.to(device)).argmax(1).cpu().tolist()
        ys += y.tolist()
    return ys, ps

def report(y, p, labels, save_path=None):
    # optionally save a confusion matrix image
    if save_path:
        ConfusionMatrixDisplay(confusion_matrix(y, p), display_labels=labels).plot(xticks_rotation=45)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
    # accuracy = percent correct, macro F1 = average score across the 6 LLMs
    return {"accuracy": accuracy_score(y, p), "macro_f1": f1_score(y, p, average="macro")}