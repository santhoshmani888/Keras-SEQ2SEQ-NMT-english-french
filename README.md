# Keras-SEQ2SEQ-NMT-english-french using pretrained word vectors
encoder-decoder model using LSTM
1.english to french using pretrained GLOVE (100d) vectors
2.French to english using pretrained word2vec vectors(200d)

Dataset:
English to French sentence pairs. http://www.manythings.org/anki/fra-eng.zip

Pre-trained word embeddings:
1.English: the 100-dimensional GloVe (https://nlp.stanford.edu/projects/glove/) embeddings of 400k words computed on a 2014 dump of English Wikipedia. 
French: the 200-dimensional frWac2Vec(http://fauconnier.github.io/ )embeddings computed using Word2vec skip-gram approach on a 1.6 billion word corpus constructed from the Web limiting the crawl to the .fr domain.

Dependencies :

    Python 3.6
    Scikit-learn, Pandas, NumPy, Matplotlib
    Keras >2.0
    Either Theano or Tensorflow backend
    
 Hyperparameters:
    batch_size = 64  # Batch size for training.
    epochs = 100  # Number of epochs to train for.
    Hidden units = 256  # Latent dimensionality of the encoding space.
    num_samples = 10000  # Number of samples to train on.
    max_vocab size = 10000
    
 English to Fr -model Summary
    __________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
encoder_inputs (InputLayer)     (None, None, 100)    0                                            
__________________________________________________________________________________________________
decoder_inputs (InputLayer)     (None, None, 116)    0                                            
__________________________________________________________________________________________________
encoder_lstm (LSTM)             [(None, 256), (None, 365568      encoder_inputs[0][0]             
__________________________________________________________________________________________________
decoder_lstm (LSTM)             [(None, None, 256),  381952      decoder_inputs[0][0]             
                                                                 encoder_lstm[0][1]               
                                                                 encoder_lstm[0][2]               
__________________________________________________________________________________________________
decoder_dense (Dense)           (None, None, 116)    29812       decoder_lstm[0][0]               
==================================================================================================
Total params: 777,332
Trainable params: 777,332
Non-trainable params: 0


Extensions working on:
1.Data Cleaning.
2.More Data
3.Layers
4.Attention

