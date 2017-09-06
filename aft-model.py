#!/usr/bin/env python3

import keras
from keras import layers
from keras.models import Model
import numpy as np

tracks = layers.Input(shape=(None, 4))
tdd_tracks = layers.TimeDistributed(layers.Dense(4))(tracks)
rnnip_raw = layers.GRU(5, return_sequences=True)(tdd_tracks)
model = Model(inputs=[tracks],
              outputs=[rnnip_raw])
model.compile(optimizer='adam', loss='categorical_crossentropy')

with open('tdd-architecture.json','w') as architecture:
    architecture.write(model.to_json(indent=2, sort_keys=True) )

model.save_weights('tdd-weights.h5')

from keras.utils.vis_utils import model_to_dot
model_to_dot(model).write_pdf('ftag-model.pdf')

trk = np.linspace(-1, 1, 20)[:,None] * np.linspace(-1, 1, 4)[None,:]
print(model.predict([trk[None,:]]))

