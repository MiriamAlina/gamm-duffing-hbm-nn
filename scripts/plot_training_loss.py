import joblib
import matplotlib.pyplot as plt
from src.util import check_folder_structure
from src.plotting import loss_plot


###############################################################################
# Check folder structure
###############################################################################
check_folder_structure()

NN_id = '2026-04-14_12-00-35'
loaded_loss = joblib.load("models/duffing_losses_h3_"+NN_id+".joblib")

train_data_losses = loaded_loss['train_losses']
validation_losses = loaded_loss['validation_losses']

figure_name = f'losses_Duffing_H3_{NN_id}'
loss_plot(train_data_losses, validation_losses, figure_name)

plt.show()
