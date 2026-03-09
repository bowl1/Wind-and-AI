import torch
import torch.nn as nn
import numpy as np
import copy


def train_streaming(
    model,
    train_loader,
    val_loader,
    device,
    epochs=10,
    lr=1e-3,
    patience=5,
    y_scaler=None,
):

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    best_val = float("inf")
    best_state = None
    patience_counter = 0

    for epoch in range(epochs):

        model.train()

        preds = []
        targets = []

        for x, y in train_loader:

            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            pred = model(x, None, device, None)
            loss = criterion(pred, y)

            loss.backward()
            optimizer.step()

            preds.append(pred.detach().cpu())
            targets.append(y.detach().cpu())

        if len(preds) == 0:
            print("⚠️ No training data!")
            continue

        train_preds = torch.cat(preds).numpy()
        train_targets = torch.cat(targets).numpy()

        # inverse transform to original scale if scaler provided
        if y_scaler is not None:
            train_preds = y_scaler.inverse_transform(train_preds)
            train_targets = y_scaler.inverse_transform(train_targets)

        train_mse = np.mean((train_preds - train_targets) ** 2)
        train_rmse = np.sqrt(train_mse)
        train_mae = np.mean(np.abs(train_preds - train_targets))

        ss_res = np.sum((train_targets - train_preds) ** 2)
        ss_tot = np.sum((train_targets - np.mean(train_targets)) ** 2)

        train_r2 = 0.0 if ss_tot < 1e-8 else 1 - ss_res / ss_tot

        val_mse, val_rmse, val_mae, val_r2, _ = test_streaming(
            model,
            val_loader,
            device,
            y_scaler=y_scaler,
        )

        print(
            f"\nEpoch {epoch+1}/{epochs}"
            f"\nTrain → MSE:{train_mse:.4f} RMSE:{train_rmse:.4f} "
            f"MAE:{train_mae:.4f} R2:{train_r2:.4f}"
            f"\nVal   → MSE:{val_mse:.4f} RMSE:{val_rmse:.4f} "
            f"MAE:{val_mae:.4f} R2:{val_r2:.4f}"
        )

        # early stopping
        if val_mse < best_val:
            best_val = val_mse
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print("🛑 Early stopping triggered")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model



def test_streaming(model, data_loader, device, y_scaler=None):

    model.eval()

    preds = []
    targets = []

    with torch.no_grad():
        for x, y in data_loader:
            x = x.to(device)
            y = y.to(device)

            pred = model(x, None, device, None)

            preds.append(pred.cpu())
            targets.append(y.cpu())

    if len(preds) == 0:
        print("⚠️ No validation data!")
        return 0, 0, 0, 0, 0

    preds = torch.cat(preds).numpy()
    targets = torch.cat(targets).numpy()

    # inverse transform to original scale if scaler provided (matches federated evaluation)
    if y_scaler is not None:
        preds = y_scaler.inverse_transform(preds)
        targets = y_scaler.inverse_transform(targets)

    mse = np.mean((preds - targets) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(preds - targets))

    ss_res = np.sum((targets - preds) ** 2)
    ss_tot = np.sum((targets - np.mean(targets)) ** 2)

    if ss_tot < 1e-8:
        r2 = 0.0
    else:
        r2 = 1 - ss_res / ss_tot

    nrmse = rmse / (np.max(targets) - np.min(targets) + 1e-8)

    return mse, rmse, mae, r2, nrmse