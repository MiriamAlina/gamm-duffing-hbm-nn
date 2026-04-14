import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import joblib
from src.util import check_folder_structure


###############################################################################
# Check folder structure
###############################################################################
check_folder_structure()


###############################################################################
# hyperparameters
###############################################################################
SAVE = True  # whether to save the trained model
if not SAVE:
    RED = "\033[91m"
    RESET = "\033[0m"
    input(f'{RED}WARNING: SAVE is set to {SAVE}. To save the model, ' +
          f'set SAVE = True. Press Enter to continue...{RESET}')
data_id = '2026-02-18_14-04-47'
n_epochs = 1000
batch_size = 64

###############################################################################
# load data
###############################################################################
train_data = np.load(f'data/duffing_train_data_H3_N64_{data_id}.npz')
X_train = train_data['q_coeffs']  # input: [a1, b1, a3, b3]
y_train = train_data['fnl_coeffs']  # output: [A1, B1, A3, B3]

val_data = np.load(f'data/duffing_val_data_H3_N64_{data_id}.npz')
X_val = val_data['q_coeffs']
y_val = val_data['fnl_coeffs']

###############################################################################
# compute scaling parameters from training data only
###############################################################################
X_mean = X_train.mean(axis=0)
X_std = X_train.std(axis=0)
X_std[X_std == 0] = 1.0

y_mean = y_train.mean(axis=0)
y_std = y_train.std(axis=0)
y_std[y_std == 0] = 1.0

###############################################################################
# scale data
###############################################################################
X_train = (X_train - X_mean) / X_std
X_val = (X_val - X_mean) / X_std

y_train = (y_train - y_mean) / y_std
y_val = (y_val - y_mean) / y_std

###############################################################################
# convert to PyTorch tensors
###############################################################################
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)

X_val = torch.tensor(X_val, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.float32)

train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

print('Data shapes:')
print(f'X_train.shape: {X_train.shape}, y_train.shape: {y_train.shape}')
print(f'X_val.shape: {X_val.shape}, y_val.shape: {y_val.shape}')

###############################################################################
# define and train the model
###############################################################################
model = nn.Sequential(
    nn.Linear(4, 128),
    nn.GELU(),
    nn.Linear(128, 128),
    nn.GELU(),
    nn.Linear(128, 128),
    nn.GELU(),
    nn.Linear(128, 4)
)

loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.003)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.5,
    patience=20,
    threshold=1e-4,
    threshold_mode='rel',
    min_lr=1e-8
)

train_data_losses = []
validation_losses = []

best_val_loss = float('inf')
best_model_state = None
epochs_without_improvement = 0
early_stopping_patience = 50

print('\nStarting training:')
for epoch in range(n_epochs):
    model.train()
    epoch_data_loss = 0.0
    n_batches = 0

    for Xbatch, ybatch in train_loader:
        optimizer.zero_grad()
        y_pred = model(Xbatch)
        data_loss = loss_fn(y_pred, ybatch)
        data_loss.backward()
        optimizer.step()

        epoch_data_loss += data_loss.item()
        n_batches += 1

    train_data_loss = epoch_data_loss / n_batches
    train_data_losses.append(train_data_loss)

    model.eval()
    with torch.no_grad():
        y_validation_pred = model(X_val)
        validation_loss = loss_fn(y_validation_pred, y_val).item()

    validation_losses.append(validation_loss)
    scheduler.step(validation_loss)

    if validation_loss < best_val_loss:
        best_val_loss = validation_loss
        best_model_state = copy.deepcopy(model.state_dict())
        epochs_without_improvement = 0
    else:
        epochs_without_improvement += 1

    current_lr = optimizer.param_groups[0]['lr']
    print(
        f'Epoch {epoch:4d} | '
        f'train loss = {train_data_loss:.3e} |'
        f'val loss = {validation_loss:.3e} | '
        f'lr = {current_lr:.3e}'
    )

    if epochs_without_improvement >= early_stopping_patience:
        print(f'Early stopping at epoch {epoch+1}')
        break

if best_model_state is not None:
    model.load_state_dict(best_model_state)

print('\nFinished training:')
print(f'Final train loss: {train_data_losses[-1]:.3e}')
print(f'Final validation loss: {validation_losses[-1]:.3e}')

###############################################################################
# save model and training history
###############################################################################
if SAVE:
    # save current date to be able to load the model later
    save_date = np.datetime64('now').astype('str').replace(
        ':', '-').replace('T', '_')
    torch.save(model, f'models/duffing_mlp_h3_{save_date}.pt')
    joblib.dump({'train_losses': train_data_losses,
                 'validation_losses': validation_losses},
                f'models/duffing_losses_h3_{save_date}.joblib')
    joblib.dump({
        'X_mean': X_mean,
        'X_std': X_std,
        'y_mean': y_mean,
        'y_std': y_std
    }, f'models/duffing_scaler_h3_{save_date}.joblib')
    print(f'Model and scaler saved with date id {save_date}')

print('\nTrained model:')
print(model)
