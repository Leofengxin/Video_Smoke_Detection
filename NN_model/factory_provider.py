import logging
from NN_model.cnn3d import Cnn3d, Cnn3d_conf
from NN_model.cnn2d_lstm import Cnn2d_lstm, Cnn2d_lstm_conf

class model_factory(object):
    models = {}
    models['cnn3d'] = Cnn3d
    models['cnn2d_lstm'] = Cnn2d_lstm
    def get_model(self, model_name, hparams):
        if model_name in self.models:
            model = self.models[model_name](hparams)
            logging.info("Model (type):{} is created successfully!".format(model_name))
            return model
        else:
            logging.error("Model is not existing! Please check your code!")
            return None

class hparams_factory(object):
    models = {}
    models['cnn3d'] = Cnn3d_conf()
    models['cnn2d_lstm'] = Cnn2d_lstm_conf()
    def get_model_hparams(self, model_name):
        if model_name in self.models:
            model = self.models[model_name]
            logging.info("Get model({}) hparams.".format(model_name))
            return model
        else:
            logging.error("Model({}) hparams is not existing! Please check your code!".format(model_name))
            return None