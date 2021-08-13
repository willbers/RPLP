import torch

import time
import os
import numpy as np
import pickle
import logging
logging.basicConfig(format='%(asctime)s - %(levelname)s: %(message)s',
                    level=logging.INFO)

from process.preprocess import init_embeddings, build_data
from process.create_batch import Triples
from config import RPLP_config as config


def load_data():
    time1 = time.time()
    train_data, validation_data, test_data, entity2id, relation2id, headTailSelector, unique_entities_train = build_data(
        config['data'], is_unweigted=False, directed=True)

    if config['pretrained_emb']:
        entity_embeddings, relation_embeddings = init_embeddings(os.path.join(config['data'], 'entity2vec.txt'),
                                                                 os.path.join(config['data'], 'relation2vec.txt'))
        logging.info("Relations and Entities embeddings are initialised from TransE")

    else:
        entity_embeddings = np.random.randn(len(entity2id), config['embedding_size'])
        relation_embeddings = np.random.randn(len(relation2id), config['embedding_size'])
        logging.info("Relations and Entities embeddings are initialised randomly")

    triples = Triples(train_data, validation_data, test_data, entity2id, relation2id, headTailSelector,
                      config['batch_size_gat'], config['valid_invalid_ratio_gat'], unique_entities_train,
                    entity_embeddings.shape[1], relation_embeddings.shape[1], config['get_2hop'])

    time2 = time.time()
    logging.info("Data building costs:{}".format(time2 - time1))

    return triples, torch.FloatTensor(entity_embeddings), torch.FloatTensor(relation_embeddings)

def load_nhop_data(triples):
    if (config['get_2hop']):
        file_positive = config['data'] + "/2hop_positive.pickle"
        file_negative = config['data'] + "/2hop_negative.pickle"
        with open(file_positive, 'wb') as handle:
            pickle.dump(triples.node_neighbors_2hop_positive, handle, protocol=pickle.HIGHEST_PROTOCOL)
        logging.info("2hop_positive.pickle writed!")
        with open(file_negative, 'wb') as handle:
            pickle.dump(triples.node_neighbors_2hop_negative, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return triples.node_neighbors_2hop_positive


    if (config['use_2hop']):
        logging.info("Opening node_neighbors pickle object")
        file_positive = config['data'] + "/2hop_positive.pickle"
        file_negative = config['data'] + "/2hop_negative.pickle"
        with open(file_positive, 'rb') as handle:
            node_neighbors_2hop_positive = pickle.load(handle)
        with open(file_negative, 'rb') as handle:
            node_neighbors_2hop_negative = pickle.load(handle)
        triples.node_neighbors_2hop_positive = node_neighbors_2hop_positive
        triples.node_neighbors_2hop_negative = node_neighbors_2hop_negative

        return node_neighbors_2hop_positive