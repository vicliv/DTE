from pyod.models.vae import VAE

class Vae():
    def __init__(self, seed=42, model_name="VAE", num_features=4,
                 latent_dim=2, hidden_activation='relu',
                 output_activation='sigmoid', optimizer='adam',
                 epochs=100, batch_size=32, dropout_rate=0.2,
                 l2_regularizer=0.1, validation_size=0.1, preprocessing=False,
                 verbose=1, random_state=None, contamination=0.1,
                 gamma=1.0, capacity=0.0):
        
        self.seed = seed
        self.model_name = model_name
    
        self.model = VAE(encoder_neurons=[num_features*4, num_features*2, num_features], decoder_neurons=[num_features, num_features*2, num_features*4],
                 latent_dim=2, hidden_activation=hidden_activation,
                 output_activation=output_activation, optimizer=optimizer,
                 epochs=epochs, batch_size=batch_size, dropout_rate=dropout_rate,
                 l2_regularizer=l2_regularizer, validation_size=validation_size, preprocessing=preprocessing,
                 verbose=verbose, contamination=contamination,
                 gamma=gamma, capacity=capacity)
        
    def fit(self, X_train, y_train=None):
        self.model.fit(X_train)
        
        return self
    
    def predict_score(self, X_test):
        return self.model.decision_function(X_test)