from keras.models import Model
from keras.layers import Input, Conv2D, ELU, Concatenate


class ExGan:
    def __init__(self, generator, discriminator):
        self.generator = generator
        self.discriminator = discriminator

        self._model = None
        self._build()

    def _build(self):
        # TODO
        self._model = Model(inputs=[], outputs=[])

    def fit_generator(self, batches, steps_per_epoch, epochs):
        pass

    def predict(self, batch):
        pass
