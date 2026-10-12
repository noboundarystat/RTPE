"""Training / evaluation loops shared by all neural models."""
import copy

import numpy as np
import torch
from sklearn.metrics import mean_squared_error


def unpack_batch(batch, device):
    """Batches are (features, time, pad_mask, [value_mask,] label)."""
    if len(batch) == 5:
        features, time, mask, value_mask, labels = batch
        value_mask = value_mask.to(device)
    elif len(batch) == 4:
        features, time, mask, labels = batch
        value_mask = None
    else:
        raise ValueError(f"Unexpected batch length: {len(batch)}")

    return (
        features.to(device),
        time.to(device),
        mask.to(device),
        value_mask,
        labels.view(-1).to(device),
    )


def predict(model, data_loader, device):
    """Return (predictions, labels) as flat float32 arrays."""
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for batch in data_loader:
            features, time, mask, value_mask, y = unpack_batch(batch, device)
            out = model(features, time, mask, value_mask)
            preds.append(out.detach().cpu().view(-1).numpy())
            labels.append(y.detach().cpu().view(-1).numpy())
    return np.concatenate(preds), np.concatenate(labels)


def validation_mse(model, data_loader, device):
    preds, labels = predict(model, data_loader, device)
    return mean_squared_error(labels, preds)


def train_once(model, train_loader, val_loader, criterion, optimizer, num_epochs, device,
               patience=5, min_epochs=15, improve_tol=1e-4):
    """Train with validation-based early stopping; restores the best-validation weights.

    Early stopping uses only the validation MSE. The reported train MSE is the running
    average of the mini-batch losses during the epoch (dropout active).
    """
    best_val_mse = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    no_improve = 0

    for epoch in range(num_epochs):
        model.train()
        running_loss, running_count = 0.0, 0
        for batch in train_loader:
            features, time, mask, value_mask, labels = unpack_batch(batch, device)
            optimizer.zero_grad()
            loss = criterion(model(features, time, mask, value_mask), labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            running_loss += loss.item() * labels.numel()
            running_count += labels.numel()

        train_mse = running_loss / max(running_count, 1)
        val_mse = validation_mse(model, val_loader, device)
        print(f"Epoch {epoch + 1}/{num_epochs} | Train MSE: {train_mse:.4f} | Val MSE: {val_mse:.4f}")

        if best_val_mse - val_mse > improve_tol:
            best_val_mse = val_mse
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1

        if (epoch + 1) >= min_epochs and no_improve >= patience:
            print(f"Early stopping at epoch {epoch + 1} (no val improvement for {patience} epochs).")
            break

    model.load_state_dict(best_state)
    return best_val_mse
