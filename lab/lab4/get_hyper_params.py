def get_hyper_params():
    """
    Define here the parameters of the architecture of the model, such as the
    layers_size, the activation. Define also the parameters related to the 
    training like the learning rate, the nuber of epochs, the initialization
    for the weights and the dropout rate.
    """
    #################################
    ##### INSERT YOUR CODE HERE #####
    # Uncomment the below code and set your parameters!
    """layers_size = [128, 10]
    activation = 'sigmoid'
    lr = 8e-1
    epochs = 10
    init_kind = 'zeros'
    dropout_rate = 0.0"""
    ##### END YOUR CODE HERE ########
    #################################
    hyper_params = {
        'layers_size': layers_size,
        'activation': activation,
        'lr': lr,
        'epochs': epochs,
        'init_kind': init_kind,
        'dropout_rate': dropout_rate,
    }
    return hyper_params

# Create the model
hyper_params = get_hyper_params()
model = MLP(
    in_dim=in_dim,
    layers_size=hyper_params['layers_size'],
    activation=hyper_params['activation'],
    lr=hyper_params['lr'],
    init_kind=hyper_params['init_kind'],
    dropout_rate=hyper_params['dropout_rate'],
).to(device)

# See model architecture
print(model)