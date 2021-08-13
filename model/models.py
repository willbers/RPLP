import time
import logging
logging.basicConfig(format='%(asctime)s - %(levelname)s: %(message)s',
                    level=logging.INFO)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from model.layers import SpGraphAttentionLayer, ConvKB
from config import RPLP_config as config

CUDA = torch.cuda.is_available()

def save_model(model, epoch, folder_name):
    logging.info("Saving Model")
    model_name = (folder_name + "trained_{}.pth").format(epoch)
    torch.save(model.state_dict(), model_name)
    logging.info("Done saving Model: {}".format(model_name))

class SpGAT(nn.Module):
    def __init__(self, num_nodes, nfeat, nhid, relation_dim, dropout, alpha, nheads):
        """
            Sparse version of GAT
        """
        super(SpGAT, self).__init__()
        self.dropout = dropout
        self.dropout_layer = nn.Dropout(self.dropout)
        self.attentions = [SpGraphAttentionLayer(num_nodes, nfeat, nhid, relation_dim, dropout=dropout, alpha=alpha, concat=True)
                           for _ in range(nheads)]

        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        # W matrix to convert h_input to h_output dimension
        self.W = nn.Parameter(torch.zeros(size=(relation_dim, nheads * nhid)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        self.out_att = SpGraphAttentionLayer(num_nodes, nhid * nheads,
                                             nheads * nhid,
                                             nheads * nhid,
                                             dropout=dropout,
                                             alpha=alpha,
                                             concat=False
                                             )

    def forward(self, entity_embeddings, relation_embed, edge_list, edge_type, edge_embed,
                edge_list_nhop, edge_type_nhop, entity_embeddings_mapping):

        out_entity = entity_embeddings
        edge_embed_nhop = relation_embed[edge_type_nhop[:, 0]] + relation_embed[edge_type_nhop[:, 1]]
        edge_type_att = torch.cat([edge_type, edge_type_nhop[:, 0]], dim=0)
        out_entity_1 = self.attentions[0](out_entity, entity_embeddings_mapping, edge_list, edge_embed, edge_list_nhop, edge_embed_nhop, edge_type_att)
        out_entity_2 = self.attentions[1](out_entity, entity_embeddings_mapping, edge_list, edge_embed, edge_list_nhop, edge_embed_nhop, edge_type_att)
        out_entity = torch.cat((out_entity_1, out_entity_2), dim=1)
        out_entity = self.dropout_layer(out_entity)

        out_relation = relation_embed.mm(self.W)
        edge_embed = out_relation[edge_type]
        edge_embed_nhop = out_relation[edge_type_nhop[:, 0]] + out_relation[edge_type_nhop[:, 1]]

        out_entity = self.out_att(out_entity, entity_embeddings_mapping, edge_list, edge_embed, edge_list_nhop, edge_embed_nhop, edge_type)
        out_entity = F.elu(out_entity)

        return out_entity, out_relation


class RPLP(nn.Module):
    def __init__(self, initial_entity_emb, initial_relation_emb):
        super().__init__()

        self.num_nodes = initial_entity_emb.shape[0]
        self.entity_in_dim = initial_entity_emb.shape[1]
        self.entity_out_dim_1 = config['entity_out_dim'][0]
        self.nheads_GAT_1 = config['nheads_GAT'][0]
        self.entity_out_dim_2 = config['entity_out_dim'][1]
        self.nheads_GAT_2 = config['nheads_GAT'][1]

        # Properties of Relations
        self.num_relation = initial_relation_emb.shape[0]
        self.relation_dim = initial_relation_emb.shape[1]
        self.relation_out_dim_1 = config['relation_out_dim'][0]

        self.drop_GAT = config['drop_GAT']
        self.alpha = config['alpha']  # For leaky relu

        self.final_entity_embeddings = nn.Parameter(
            torch.randn(self.num_nodes, self.entity_out_dim_1 * self.nheads_GAT_1))

        self.final_relation_embeddings = nn.Parameter(
            torch.randn(self.num_relation, self.entity_out_dim_1 * self.nheads_GAT_1))

        self.entity_embeddings = nn.Parameter(initial_entity_emb)
        self.relation_embeddings = nn.Parameter(initial_relation_emb)

        self.sparse_gat_1 = SpGAT(self.num_nodes, self.entity_in_dim, self.entity_out_dim_1, self.relation_dim,
                                  self.drop_GAT, self.alpha, self.nheads_GAT_1)

        self.W_entities = nn.Parameter(torch.zeros(
            size=(self.entity_in_dim, self.entity_out_dim_1 * self.nheads_GAT_1)))
        nn.init.xavier_uniform_(self.W_entities.data, gain=1.414)

        # entity -> relation mapping matrix，after update
        self.total_trans_matrix_after_update = nn.Parameter(torch.zeros(
            size=(
            self.num_relation, self.entity_out_dim_1 * self.nheads_GAT_1, self.entity_out_dim_1 * self.nheads_GAT_1)))

        self.final_trans_matrix_after_update = nn.Parameter(torch.Tensor(size=(
        self.num_relation, self.entity_out_dim_1 * self.nheads_GAT_1, self.entity_out_dim_1 * self.nheads_GAT_1)))

        # entity -> relation mapping matrix，before update
        self.total_trans_matrix_before_update = nn.Parameter(torch.zeros(
            size=(
            self.num_relation, self.entity_in_dim, self.entity_in_dim)))

        self.final_trans_matrix_before_update = nn.Parameter(torch.Tensor(size=(
            self.num_relation, self.entity_in_dim, self.entity_in_dim)))

    def forward(self, triples, adj, batch_inputs, train_indices_nhop, flag='positive'):
        # getting edge list
        edge_list = adj[0]
        edge_type = adj[1]
        CUDA = torch.cuda.is_available()

        if flag == 'negative':
            train_indices_nhop = triples.get_nhop_neighbors_attention(triples.unique_entities_train,
                                                                                   triples.node_neighbors_2hop_negative,
                                                                                   self.entity_embeddings,
                                                                                   self.relation_embeddings)
            if CUDA:
                train_indices_nhop = Variable(
                    torch.LongTensor(train_indices_nhop)).cuda()
            else:
                train_indices_nhop = Variable(
                    torch.LongTensor(train_indices_nhop))

        edge_list_nhop = torch.cat(
            (train_indices_nhop[:, 3].unsqueeze(-1), train_indices_nhop[:, 0].unsqueeze(-1)), dim=1).t()
        edge_type_nhop = torch.cat(
            [train_indices_nhop[:, 1].unsqueeze(-1), train_indices_nhop[:, 2].unsqueeze(-1)], dim=1)

        if(CUDA):
            edge_list = edge_list.cuda()
            edge_type = edge_type.cuda()
            edge_list_nhop = edge_list_nhop.cuda()
            edge_type_nhop = edge_type_nhop.cuda()

        edge_embed = self.relation_embeddings[edge_type]

        self.entity_embeddings.data = F.normalize(
            self.entity_embeddings.data, p=2, dim=1).detach()

        # Add mapping matrics before updating
        entity_embeddings_mapping = torch.matmul(self.entity_embeddings, self.total_trans_matrix_before_update)

        out_entity_1, out_relation_1 = self.sparse_gat_1(
            self.entity_embeddings, self.relation_embeddings, edge_list, edge_type, edge_embed, edge_list_nhop,
            edge_type_nhop, entity_embeddings_mapping)
        # print("sparse gat done!")

        mask_indices = torch.unique(batch_inputs[:, 2]).cuda()
        mask = torch.zeros(self.entity_embeddings.shape[0]).cuda()
        mask[mask_indices] = 1.0

        entities_upgraded = self.entity_embeddings.mm(self.W_entities)
        out_entity_1 = entities_upgraded + mask.unsqueeze(-1).expand_as(out_entity_1) * out_entity_1

        out_entity_1 = F.normalize(out_entity_1, p=2, dim=1)

        # Add mapping matrics after updating
        out_entity_relation = torch.matmul(out_entity_1, self.total_trans_matrix_after_update)

        self.final_trans_matrix_after_update = self.total_trans_matrix_after_update
        self.final_trans_matrix_before_update = self.total_trans_matrix_before_update

        return out_entity_1, out_relation_1, out_entity_relation


class ConvOnly(nn.Module):
    def __init__(self, initial_entity_emb, initial_relation_emb):
        super().__init__()

        self.num_nodes = initial_entity_emb.shape[0]
        self.entity_in_dim = initial_entity_emb.shape[1]
        self.entity_out_dim_1 = config['entity_out_dim'][0]
        self.nheads_GAT_1 = config['nheads_GAT'][0]
        self.entity_out_dim_2 = config['entity_out_dim'][1]
        self.nheads_GAT_2 = config['nheads_GAT'][1]

        # Properties of Relations
        self.num_relation = initial_relation_emb.shape[0]
        self.relation_dim = initial_relation_emb.shape[1]
        self.relation_out_dim_1 = config['relation_out_dim'][0]

        self.drop_GAT = config['drop_GAT']
        self.drop_conv = config['drop_conv']
        self.alpha = config['alpha']      # For leaky relu
        self.alpha_conv = config['alpha_conv']
        self.conv_out_channels = config['conv_out_channels']

        self.final_entity_embeddings = nn.Parameter(
            torch.randn(self.num_nodes, self.entity_out_dim_1 * self.nheads_GAT_1))

        self.final_relation_embeddings = nn.Parameter(
            torch.randn(self.num_relation, self.entity_out_dim_1 * self.nheads_GAT_1))

        in_channels = self.entity_out_dim_1 * self.nheads_GAT_1
        out_channels = self.entity_out_dim_1 * self.nheads_GAT_1
        shape = (self.num_relation, in_channels, out_channels)
        self.weight_matrices = nn.Parameter(nn.init.xavier_normal_(nn.Parameter(
            torch.Tensor(*shape)).data)).cuda()
        self.final_weight_matrices = nn.Parameter(torch.Tensor(*shape))

        self.convKB = ConvKB(self.entity_out_dim_1 * self.nheads_GAT_1, 3, 1,
                             self.conv_out_channels, self.drop_conv, self.alpha_conv)

    def proj(self, embeddings, matrices):
        embeddings = embeddings.unsqueeze(1)
        return torch.bmm(embeddings, matrices).squeeze(1)

    def forward(self, batch_inputs):

        projected_head = self.proj(self.final_entity_embeddings[batch_inputs[:, 0], :], self.weight_matrices[batch_inputs[:, 1], :, :])
        projected_tail = self.proj(self.final_entity_embeddings[batch_inputs[:, 2], :], self.weight_matrices[batch_inputs[:, 1], :, :])
        conv_input = torch.cat((projected_head.unsqueeze(1),
                                self.final_relation_embeddings[batch_inputs[:, 1]].unsqueeze(1),
                                projected_tail.unsqueeze(1)), dim=1)
        out_conv = self.convKB(conv_input)
        self.final_weight_matrices = self.weight_matrices
        return out_conv

    def batch_test(self, batch_inputs):

        projected_head = self.proj(self.final_entity_embeddings[batch_inputs[:, 0], :], self.final_weight_matrices[batch_inputs[:, 1], :, :])
        projected_tail = self.proj(self.final_entity_embeddings[batch_inputs[:, 2], :], self.final_weight_matrices[batch_inputs[:, 1], :, :])

        conv_input = torch.cat((projected_head.unsqueeze(1), self.final_relation_embeddings[
            batch_inputs[:, 1]].unsqueeze(1), projected_tail.unsqueeze(1)), dim=1)
        out_conv = self.convKB(conv_input)
        return out_conv
