import torch
import torch.nn as nn
from torch.autograd import Variable

import random
import numpy as np
import time
import logging
logging.basicConfig(format='%(asctime)s - %(levelname)s: %(message)s',
                    level=logging.INFO)

from model.models import RPLP, ConvOnly, save_model
from model.loss import batch_loss
from process.metrics import validation
from config import RPLP_config as config

CUDA = torch.cuda.is_available()

def train_gat(triples, entity_embeddings, relation_embeddings, nhop_nodes):

    # Creating the gat model here.
    ####################################

    logging.info("Defining model")
    model_RPLP = RPLP(entity_embeddings, relation_embeddings)

    if CUDA:
        model_RPLP.cuda()

    optimizer = torch.optim.Adam(model_RPLP.parameters(), lr=config['lr'], weight_decay=config['weight_decay_gat'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.5, last_epoch=-1)
    loss_func = nn.MarginRankingLoss(margin=config['margin'])

    if(config['use_2hop']):
        nhop_indices_positive = triples.get_nhop_neighbors(triples.unique_entities_train, nhop_nodes)

    if CUDA:
        current_nhop_indices_positive = Variable(torch.LongTensor(nhop_indices_positive)).cuda()
    else:
        current_nhop_indices_positive = Variable(torch.LongTensor(nhop_indices_positive))

    epoch_losses = []   # losses of all epochs
    logging.info("Number of gat epochs: {}".format(config['epochs_gat']))
    logging.info(("=" * 50))
    logging.info(("=" * 16 + 'Begin gat training' +"=" * 16))
    logging.info(("=" * 50))
    logging.info("Nums of entities in training: {}".format(len(triples.unique_entities_train)))
    for epoch in range(config['epochs_gat']):
        logging.info(("=" * 25))
        logging.info("epoch: {}".format(epoch))
        random.shuffle(triples.train_triples)
        triples.train_indices = np.array(list(triples.train_triples)).astype(np.int32)
        model_RPLP.train()  # getting in training mode
        start_time = time.time()
        epoch_loss = []

        if len(triples.train_indices) % config['batch_size_gat'] == 0:
            num_iters_per_epoch = len(triples.train_indices) // config['batch_size_gat']
        else:
            num_iters_per_epoch = (len(triples.train_indices) // config['batch_size_gat']) + 1

        for iters in range(num_iters_per_epoch):
            train_indices, train_values = triples.get_iteration_batch(iters)

            if CUDA:
                train_indices = Variable(torch.LongTensor(train_indices)).cuda()
            else:
                train_indices = Variable(torch.LongTensor(train_indices))

            # forward
            entity_embed_positive, relation_embed_positive, entity_relation_embed_positive = model_RPLP(
                triples, triples.train_adj_matrix, train_indices, train_indices_nhop=current_nhop_indices_positive, flag='positive')

            model_RPLP.final_entity_embeddings.data = entity_embed_positive.data
            model_RPLP.final_relation_embeddings.data = relation_embed_positive.data

            entity_embed_negative, relation_embed_negative, entity_relation_embed_negative = model_RPLP(
                triples, triples.train_adj_matrix, train_indices, train_indices_nhop='', flag='negative')

            model_RPLP.final_entity_embeddings.data = entity_embed_negative.data
            model_RPLP.final_relation_embeddings.data = relation_embed_negative.data

            optimizer.zero_grad()

            loss = batch_loss(loss_func, train_indices, relation_embed_negative, entity_relation_embed_negative)

            loss.backward()
            optimizer.step()

            epoch_loss.append(loss.data.item())

        scheduler.step()
        logging.info("Epoch: {} , average loss {} , epoch_time: {}".format(
                epoch, sum(epoch_loss) / len(epoch_loss), time.time() - start_time))
        if (epoch+1) % 10 == 0:
            save_model(model_RPLP, epoch, config['output_folder'])

        epoch_losses.append(sum(epoch_loss) / len(epoch_loss))


def train_conv(triples, entity_embeddings, relation_embeddings):

    # Creating convolution model here.
    ####################################

    logging.info("GAT Initialized")
    model_RPLP = RPLP(entity_embeddings, relation_embeddings)
    logging.info("Conv model for training")
    model_conv = ConvOnly(entity_embeddings, relation_embeddings)

    if CUDA:
        model_RPLP.cuda()
        model_conv.cuda()

    model_RPLP.load_state_dict(torch.load('{}/trained_{}.pth'.format(config['output_folder'], config['epochs_gat'] - 1)), strict=False)
    model_conv.final_entity_embeddings = model_RPLP.final_entity_embeddings
    model_conv.final_relation_embeddings = model_RPLP.final_relation_embeddings
    model_conv.weight_matrices = model_RPLP.total_trans_matrix_after_update

    triples.batch_size = config['batch_size_conv']
    triples.invalid_valid_ratio = int(config['valid_invalid_ratio_conv'])

    optimizer = torch.optim.Adam(model_conv.parameters(), lr=config['lr'], weight_decay=config['weight_decay_conv'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.5, last_epoch=-1)
    margin_loss = torch.nn.SoftMarginLoss()

    epoch_losses = []   # losses of all epochs
    logging.info("Number of conv epochs {}".format(config['epochs_conv']))
    logging.info(("=" * 49))
    logging.info(("=" * 15 + 'Begin conv training' +"=" * 15))
    logging.info(("=" * 49))
    for epoch in range(config['epochs_conv']):
        logging.info("=" * 25)
        logging.info("Epoch: {}".format(epoch))
        random.shuffle(triples.train_triples)
        triples.train_indices = np.array(list(triples.train_triples)).astype(np.int32)

        model_conv.train()  # getting in training mode
        start_time = time.time()
        epoch_loss = []

        if len(triples.train_indices) % config['batch_size_conv'] == 0:
            num_iters_per_epoch = len(triples.train_indices) // config['batch_size_conv']
        else:
            num_iters_per_epoch = (len(triples.train_indices) // config['batch_size_conv']) + 1

        for iters in range(num_iters_per_epoch):
            train_indices, train_values = triples.get_iteration_batch(iters)

            if CUDA:
                train_indices = Variable(
                    torch.LongTensor(train_indices)).cuda()
                train_values = Variable(torch.FloatTensor(train_values)).cuda()

            else:
                train_indices = Variable(torch.LongTensor(train_indices))
                train_values = Variable(torch.FloatTensor(train_values))

            preds = model_conv(train_indices)

            optimizer.zero_grad()

            loss = margin_loss(preds.view(-1), train_values.view(-1))

            loss.backward()
            optimizer.step()

            epoch_loss.append(loss.data.item())

        scheduler.step()
        logging.info("Epoch {} , average loss {} , epoch_time {}".format(
                epoch, sum(epoch_loss) / len(epoch_loss), time.time() - start_time))
        if (epoch+1) % 10 == 0:
            save_model(model_conv, epoch, config['output_folder'] + "conv/")

        epoch_losses.append(sum(epoch_loss) / len(epoch_loss))


def evaluate(triples, entity_embeddings, relation_embeddings, unique_entities):
    model_conv = ConvOnly(entity_embeddings, relation_embeddings)
    model_conv.load_state_dict(torch.load(
        '{}conv/trained_{}.pth'.format(config['output_folder'], config['epochs_conv'] - 1)), strict=False)
    model_conv.cuda()
    model_conv.eval()
    with torch.no_grad():
        validation(triples, model_conv, unique_entities)