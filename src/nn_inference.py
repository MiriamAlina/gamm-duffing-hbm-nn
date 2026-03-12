import torch


def evaluate_Duffing_nn_H3(NN_id, X):
    model_path = f'models/duffing_mlp_h3_{NN_id}.pt'
    NN_model = torch.load(model_path, weights_only=False)
    NN_model.eval()

    nn_input = torch.tensor(X, dtype=torch.float32)

    with torch.no_grad():
        output = NN_model(nn_input)

    return output.detach().numpy()
