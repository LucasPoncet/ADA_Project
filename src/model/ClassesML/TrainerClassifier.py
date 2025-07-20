import torch
from sklearn.metrics import f1_score,accuracy_score

class TrainerClassifier:
    def __init__(self, model, optimizer, loss_fn, device, patience=5):
        self.model, self.opt, self.crit = model, optimizer, loss_fn
        self.device = device
        self.patience_init = patience
        

    def run(self, X_train, y_train, X_val, y_val, epochs):
        X_train, y_train = X_train.to(self.device), y_train.to(self.device)
        X_val,   y_val   = X_val.to(self.device),   y_val.to(self.device)

        best_f1, patience = 0, self.patience_init
        best_state = self.model.state_dict()
        train_f1_hist, val_f1_hist = [], []

        for epoch in range(1, epochs+1):
            # train 
            self.model.train()
            self.opt.zero_grad()
            logits = self.model(X_train)
            loss   = self.crit(logits, y_train)
            loss.backward()
            self.opt.step()

            preds_train = logits.argmax(dim=1).cpu().numpy()
            train_f1 = f1_score(y_train.cpu().numpy(), preds_train, average="macro")
            train_acc = accuracy_score(y_train.cpu().numpy(), preds_train)
            

            # val 
            self.model.eval()
            with torch.no_grad():
                preds_val = self.model(X_val).argmax(dim=1).cpu().numpy()
            val_f1 = f1_score(y_val.cpu().numpy(), preds_val, average="macro")
            acc_val = accuracy_score(y_val.cpu().numpy(), preds_val)
            
            train_f1_hist.append(train_f1)
            val_f1_hist.append(val_f1)
           
           
            print(
                f"epoch {epoch:02d}  loss {loss.item():.4f}  "
                f"train-F1 {train_f1:.3f}  val-F1 {val_f1:.3f}  "
                f"train-Acc {train_acc:.3f}  val-Acc {acc_val:.3f}"
            )
            # early-stop
            if val_f1 > best_f1:
                best_f1, patience = val_f1, self.patience_init
                best_state = self.model.state_dict()
            else:
                patience -= 1
                if patience == 0:
                    print("Early-stop triggered")
                    break

        # Reload best model
        self.model.load_state_dict(best_state)
        return train_f1_hist, val_f1_hist, best_f1
