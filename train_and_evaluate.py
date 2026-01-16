import torch
import torch.nn as nn
from torchmetrics.classification import BinaryAccuracy, BinaryPrecision, BinaryRecall, BinaryF1Score, BinaryConfusionMatrix
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tqdm.auto import tqdm
import pandas as pd

device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

class EarlyStopping():
    
    def __init__(self, patience=3, delta=0.01, verbose=True):
        
        self.patience = patience
        self.delta = delta
        self.verbose=verbose
        self.best_loss=None
        self.no_improvement_count=0
        self.stop_training=False
    
    def check_early_stop(self, val_loss):

        if self.best_loss is None or val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.no_improvement_count = 0
        else:
            self.no_improvement_count =+1
            if self.no_improvement_count >= self.patience:
                self.stop_training=True
                if self.verbose:
                    print("Stopping early as no improvment was observed")

early_stopping = EarlyStopping()

def train_and_test(model, train_dl, test_dl, loss_fn, optimizer, epochs):

    torch.manual_seed(42)
    accuracy_metric = BinaryAccuracy().to(device)
    precision_metric = BinaryPrecision().to(device)
    recall_metric = BinaryRecall().to(device)
    f1_metric = BinaryF1Score().to(device)
    
    for i, epoch in enumerate(range(epochs)):

        print(f"\nEpoch: {epoch+1}/{epochs}")
        train_loss = []
        model.train()

       # Before the test loop
        accuracy_metric.reset()
        precision_metric.reset()
        recall_metric.reset()
        f1_metric.reset()
       
        for batch, (X_train, y_train, idx) in tqdm(enumerate(train_dl), total=len(train_dl)):
            
            X_train, y_train = X_train.to(device, dtype=torch.float32), y_train.to(device, dtype=torch.float32)

            assert not torch.isnan(X_train).any()
            assert not torch.isnan(y_train).any()


            y_pred_with_logits = model(X_train)
            y_pred = torch.round(torch.sigmoid(y_pred_with_logits))  # Get probabilities

            loss = loss_fn(y_pred_with_logits, y_train)
            train_loss.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch % 1000 == 0:
                print(f"Seen {batch * len(X_train)} / {len(train_dl.dataset)} samples")
 
        # Test metrics instantiation
        model.eval()

        all_preds = []
        all_labels = []
        all_idx = [] 
        test_loss = []

        with torch.inference_mode():
            for X_test, y_test, idx in test_dl:

                assert not torch.isnan(X_test).any()
                assert not torch.isnan(y_test).any()

                X_test, y_test = X_test.to(device, dtype=torch.float32),y_test.to(device, dtype=torch.float32)

                y_pred_with_logits = model(X_test)
                probs = torch.sigmoid(y_pred_with_logits)
                preds = torch.round(probs)
                loss_tmp = loss_fn(y_pred_with_logits, y_test)
                test_loss.append(loss_tmp.item())

            

                all_preds.append(preds)
                all_labels.append(y_test)
                all_idx.append(idx)

                accuracy_metric.update(preds, y_test)
                recall_metric.update(preds, y_test)
                f1_metric.update(preds,y_test)
                precision_metric.update(preds, y_test)


        # Test metrics 
        test_loss_avg = sum(test_loss) / len(test_loss)
        acc = accuracy_metric.compute()
        rec = recall_metric.compute()
        f1 = f1_metric.compute()
        prec = precision_metric.compute()
        
        print(f"Test loss: {test_loss_avg:.5f}")
        print(f"Accuracy: {acc:.3f}")
        print(f"Precision: {prec:.3f} | Recall: {rec:.3f} | F1: {f1:.3f}")

        # Early stopping call
        early_stopping.check_early_stop(test_loss_avg)
            
        if early_stopping.stop_training:
            print(f"Early stopping at epoch {epoch}")
            break



        if i % 10 == 0:

            train_loss_avg = sum(train_loss) / len(train_loss)
            print(f"Train loss: {train_loss_avg:.5f}")

            #  Train loss
            fig_loss, ax_loss = plt.subplots(figsize=(8,6))
            ax_loss.plot(train_loss)
            ax_loss.set_title(f"Epoch {epoch+1} Train Loss")
            ax_loss.set_xlabel("Batch")
            ax_loss.set_ylabel("Loss")
            plt.show() # Always use plt.show() to display current figure

            # Confusion matrix
            all_preds_np = torch.cat(all_preds).cpu().numpy()
            all_labels_np = torch.cat(all_labels).cpu().numpy()
            cm = confusion_matrix(all_labels_np, all_preds_np)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['non-ORIV', 'ORIV'])
            fig_cm, ax_cm = plt.subplots(figsize=(8,8))
            disp.plot(ax=ax_cm)
            plt.show()



        if i == epochs-1:
            fig_cm.savefig("ConfusionMatrix.png")
            metrics = {
                "Accuracy": [acc.item()], 
                "Precision": [prec.item()], 
                "Recall": [rec.item()], 
                "F1": [f1.item()]
            }
            df = pd.DataFrame(data=metrics)
            df.to_csv("CNN_metrics.csv", index=False)
    
    # Return final predictions for test set WITH INDICES
        df = pd.DataFrame({
        'predictions': torch.cat(all_preds).cpu().numpy().flatten(),
        'labels': torch.cat(all_labels).cpu().numpy().flatten(),
        'indices': torch.cat(all_idx).cpu().numpy().flatten(),
    })
        df.to_csv("actual_vs_prediction_testset.csv")



def evaluate(model, dataloader):
    model.eval()  # set model to evaluation mode
    all_preds = []
    all_labels = []
    all_indices = [] 
    accuracy_metric = BinaryAccuracy().to(device)
    precision_metric = BinaryPrecision().to(device)
    recall_metric = BinaryRecall().to(device)
    f1_metric = BinaryF1Score().to(device)
    
    with torch.no_grad():  # disable gradient computation
        for X, y, idx in dataloader:
            X, y = X.to(device, dtype=torch.float32), y.to(device, dtype=torch.float32)

            y_pred = model(X)
                      # Store predictions and labels

            all_preds.append(torch.round(torch.sigmoid(y_pred)))
            all_labels.append(y)
            all_indices.append(idx)

    # Concatenate all batches
    all_preds_tensor = torch.cat(all_preds)
    all_labels_tensor = torch.cat(all_labels)
    all_indices_tensor = torch.cat(all_indices)

    # Compute binary predictions
    y_pred_labels = torch.round(all_preds_tensor)

    # Compute metrics
    accuracy = accuracy_metric(y_pred_labels, all_labels_tensor)
    precision = precision_metric(y_pred_labels, all_labels_tensor)
    recall = recall_metric(y_pred_labels, all_labels_tensor)
    f1 = f1_metric(y_pred_labels, all_labels_tensor)

    # Confusion matrix
    cm = confusion_matrix(all_labels_tensor.cpu().numpy(), y_pred_labels.cpu().numpy())
    disp = ConfusionMatrixDisplay(cm, display_labels=['non-ORIV', 'ORIV'])
    disp.plot(cmap=plt.cm.Blues)
    plt.show()


    print(f"Accuracy: {accuracy:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f},")

    df = pd.DataFrame({
        'predictions': all_preds_tensor.cpu().numpy().flatten(),
        'labels': all_labels_tensor.cpu().numpy().flatten(),
        'indices': all_indices_tensor.cpu().numpy().flatten(),
    })

    df.to_csv("cnn_predictions_vs_actual.csv", index=True)
    print("Saved: cnn_predictions_vs_actual.csv")
    return 
