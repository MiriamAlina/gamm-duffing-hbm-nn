import joblib
from src.plotting import loss_plot


NN_id = '2026-02-18_13-29-30'
loaded_loss = joblib.load("models/duffing_losses_h3_"+NN_id+".joblib")

train_data_losses = loaded_loss['train_losses']
validation_losses = loaded_loss['validation_losses']

figure_name = f'losses_Duffing_H3_{NN_id}'
loss_plot(train_data_losses, validation_losses, figure_name)
