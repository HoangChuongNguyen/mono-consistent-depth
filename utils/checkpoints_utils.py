
import torch
import os


def load_model(model_list, model_name_list, optimizer, load_weights_folder):
    models = dict(zip(model_name_list, model_list))
    for n in model_name_list:
        print("Loading {} weights...".format(n))
        path = os.path.join(load_weights_folder, "{}.pth".format(n))
        if n == 'scheduler':
            models[n].load_state_dict(torch.load(path))
        else:
            models[n].load_state_dict(torch.load(path), strict=True)
    if optimizer is not None:
        try:
            # loading adam state
            optimizer_load_path = os.path.join(load_weights_folder, "adam.pth")
            if os.path.isfile(optimizer_load_path):
                print("Loading Adam weights")
                optimizer_dict = torch.load(optimizer_load_path)
                optimizer.load_state_dict(optimizer_dict)
            else:
                print("Cannot find Adam weights so Adam is randomly initialized")
        except:
            print("Cannot load Adam optimizer so Adam is randomly initialized")
    try: load_epoch = int(load_weights_folder.split("_")[-1]) 
    except: load_epoch = 10000 # Set dummy value for epoch when load the best model
    return load_epoch

def save_model(model_list, model_name_list, optimizer, log_path, epoch):
    """Save model weights to disk
    """
    if epoch is not None:
        save_folder = os.path.join(log_path, "models", "weights_{}".format(epoch))
    else:
        save_folder = os.path.join(log_path, "models", "weights")
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    models = dict(zip(model_name_list, model_list))

    for model_name, model in models.items():
        save_path = os.path.join(save_folder, "{}.pth".format(model_name))
        to_save = model.state_dict()
        torch.save(to_save, save_path)

    save_path = os.path.join(save_folder, "{}.pth".format("adam"))
    torch.save(optimizer.state_dict(), save_path)