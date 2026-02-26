import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import matplotlib
import matplotlib.pyplot as plt
import joblib

matplotlib.rcParams.update({'font.size': 12})

MODE = 'inference'  # 'train' or 'inference'
SAVE = False  # whether to save the trained model

# load training data
data_load_date = '2026-02-18_14-04-47'
model_load_date = '2026-02-18_13-29-30'
data = np.load('data/duffing_training_data_H3_N64_'+data_load_date+'.npz')
q_coeffs = data['q_coeffs']  # input: [c1, s1, c3, s3]
fnl_coeffs = data['fnl_coeffs']  # output: [c1, s1, c3, s3]
# split data into 60% train, 20% valdation and 20% test
X_tmp, X_test, y_tmp, y_test = train_test_split(
    q_coeffs, fnl_coeffs, test_size=0.2, random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(
    X_tmp, y_tmp, test_size=0.25, random_state=42
)

# Scale data
X_train_scaled = np.copy(X_train)
X_val_scaled = np.copy(X_val)
X_test_scaled = np.copy(X_test)

fig, axs = plt.subplots(4, 1, figsize=(4.5, 5.5))
axs[0].hist(X_train[:, 0], bins=20, color='#A8DADC', label='Original X_train')
axs[0].hist(X_train_scaled[:, 0], bins=20, color='#E63946',
            label='Converted X_train')
axs[0].set_xlabel(r'cosine amplitude $a_1$')
axs[0].set_title('Training input')
axs[0].legend()
axs[1].hist(X_train[:, 1], bins=20, color='#A8DADC', label='Original X_train')
axs[1].hist(X_train_scaled[:, 1], bins=20, color='#E63946',
            label='Converted X_train')
axs[1].set_xlabel(r'sine amplitude $b_1$')
axs[1].set_ylabel('Number of samples')
axs[2].hist(X_train[:, 2], bins=20, color='#A8DADC', label='Original X_train')
axs[2].hist(X_train_scaled[:, 2], bins=20, color='#E63946',
            label='Converted X_train')
axs[2].set_xlabel(r'cosine amplitude $a_3$')
axs[3].hist(X_train[:, 3], bins=20, color='#A8DADC', label='Original X_train')
axs[3].hist(X_train_scaled[:, 3], bins=20, color='#E63946',
            label='Converted X_train')
axs[3].set_xlabel(r'sine amplitude $b_3$')
plt.tight_layout()
plt.show()

fig, axs = plt.subplots(4, 1, figsize=(4.5, 6.5))
# bins = np.logspace(np.log10(0.01), np.log10(10), 20)
axs[0].hist(y_train[:, 0], bins=20, color='#A8DADC')
axs[0].set_xlabel(r'cosine amplitude $A_1$')
axs[0].set_title('Training output')
axs[1].hist(y_train[:, 1], bins=20, color='#A8DADC')
axs[1].set_xlabel(r'sine amplitude $B_1$')
axs[1].set_ylabel('Number of samples')
axs[2].hist(y_train[:, 2], bins=20, color='#A8DADC')
axs[2].set_xlabel(r'cosine amplitude $A_3$')
axs[3].hist(y_train[:, 3], bins=20, color='#A8DADC')
axs[3].set_xlabel(r'sine amplitude $B_3$')
plt.tight_layout()
plt.show()

# convert to PyTorch tensors
X_train = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_val = torch.tensor(X_val_scaled, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.float32)
X_test = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)
# print(X_train.shape, y_train.shape)
# X_train.shape = (6000, 4)  # 6000 samples, c1, s1, c3, s3
# y_train.shape = (6000, 4)  # 6000 samples, c1, s1, c3, s3

if MODE == 'train':
    model = nn.Sequential(
        nn.Linear(4, 32),
        nn.ReLU(),
        nn.Linear(32, 32),
        nn.ReLU(),
        nn.Linear(32, 32),
        nn.ReLU(),
        nn.Linear(32, 32),
        nn.ReLU(),
        nn.Linear(32, 32),
        nn.ReLU(),
        nn.Linear(32, 4)
    )

    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # physics_weight = 1.0  # ask how to express physics loss in this case

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=20,
        threshold=1e-4,
        threshold_mode='rel',
        min_lr=1e-8
    )

    n_epochs = 1000
    batch_size = 10

    train_data_losses = []
    # train_phys_losses = []
    # train_total_losses = []
    validation_losses = []

    for epoch in range(n_epochs):
        epoch_data_loss = 0.0
        # epoch_phys_loss = 0.0
        # epoch_total_loss = 0.0
        n_batches = 0

        for i in range(0, len(X_train), batch_size):
            Xbatch = X_train[i:i+batch_size]
            ybatch = y_train[i:i+batch_size]

            optimizer.zero_grad()
            y_pred = model(Xbatch)
            data_loss = loss_fn(y_pred, ybatch)
            # L2 penalty of positive B values
            # physics_loss = torch.mean(nn.functional.relu(B)**2)
            loss = data_loss  # + physics_weight * physics_loss
            loss.backward()
            optimizer.step()

            epoch_data_loss += data_loss.item()
            # epoch_phys_loss += physics_loss.item()
            # epoch_total_loss += loss.item()
            n_batches += 1

        train_data_losses.append(epoch_data_loss / n_batches)
        # train_phys_losses.append(epoch_phys_loss / n_batches)
        # train_total_losses.append(epoch_total_loss / n_batches)

        model.eval()
        with torch.no_grad():
            y_validation_pred = model(X_val)
            validation_loss = loss_fn(y_validation_pred, y_val).item()
        model.train()
        validation_losses.append(validation_loss)

        scheduler.step(validation_loss)

        # shuffle training data to avoid systematic batch artefacts
        perm = torch.randperm(len(X_train))
        X_train = X_train[perm]
        y_train = y_train[perm]

        current_lr = optimizer.param_groups[0]['lr']
        print(
            f"Epoch {epoch:4d} | "
            f"train loss = {epoch_data_loss / n_batches:.3e} |"
            f"val loss = {validation_loss:.3e} | "
            f"lr = {current_lr:.3e}")

    print("Finished training with adaptive Adam")
    print(f"Final train loss: {train_data_losses[-1]:.3e}")
    print(f"Final validation loss: {validation_losses[-1]:.3e}")

    if SAVE:
        # save current date to be able to load the model later
        save_date = np.datetime64('now').astype('str').replace(
            ':', '-').replace('T', '_')
        torch.save(model, 'models/MLP_Duffing_H3_'+save_date+'.pt')
        joblib.dump({'train_losses': train_data_losses,
                     'validation_losses': validation_losses},
                    'models/losses_Duffing_H3_'+save_date+'.joblib')
        print(f"Model and scaler saved with date id {save_date}")

    plt.semilogy(train_data_losses, label='Training loss')
    plt.semilogy(validation_losses, label='Validation loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()
    plt.show()

if MODE == 'inference':
    model = torch.load('models/MLP_Duffing_H3_'+model_load_date+'.pt',
                       weights_only=False)
    model.eval()

print(model)

# test a single sample
i = 1
X_sample = X_test[i:i+1]
model.eval()
with torch.no_grad():
    y_pred = model(X_sample)
print(f"Sample:\n{X_sample[0]}\n-> Prediction:\n{y_pred[0]}\n" +
      f"-> Ground truth:\n{y_test[i]}")

# evaluate the model on the test set
with torch.no_grad():
    y_pred = model(X_test)
mae = (y_pred - y_test).abs().mean(axis=0)
print(f"mean difference over test set: {np.round(np.asarray(mae), 4)}")
