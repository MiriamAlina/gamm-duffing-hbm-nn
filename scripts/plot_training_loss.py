import joblib
import matplotlib.pyplot as plt

NN_id = '2026-02-18_13-29-30'
loaded_loss = joblib.load("models/losses_Duffing_H3_"+NN_id+".joblib")

train_data_losses = loaded_loss['train_losses']
validation_losses = loaded_loss['validation_losses']
plt.figure(figsize=(4, 3))
plt.semilogy(train_data_losses, color='#00aebf', label='Training loss')
plt.semilogy(validation_losses, color='#e63946', label='Validation loss')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig('figures/losses_Duffing_H3_'+NN_id+'.svg', bbox_inches='tight')
plt.show()
