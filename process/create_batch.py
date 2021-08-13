import torch
import numpy as np
from collections import defaultdict
import time
import queue
import random
import logging
logging.basicConfig(format='%(asctime)s - %(levelname)s: %(message)s',
                    level=logging.INFO)
import torch
from config import RPLP_config as config

class Triples:
    def __init__(self, train_data, validation_data, test_data, entity2id,
                 relation2id, headTailSelector, batch_size, valid_to_invalid_samples_ratio,
                 unique_entities_train, entity_dim, relation_dim, get_2hop=False):
        self.train_triples = train_data[0]

        # Converting to sparse tensor
        adj_indices = torch.LongTensor(
            [train_data[1][0], train_data[1][1]])  # rows and columns
        adj_values = torch.LongTensor(train_data[1][2])
        self.train_adj_matrix = (adj_indices, adj_values)

        # adjacency matrix is needed for train_data only, as GAT is trained for
        # training data
        self.validation_triples = validation_data[0]
        self.test_triples = test_data[0]

        self.headTailSelector = headTailSelector  # for selecting random entities
        self.entity2id = entity2id
        self.id2entity = {v: k for k, v in self.entity2id.items()}
        self.relation2id = relation2id
        self.id2relation = {v: k for k, v in self.relation2id.items()}
        self.batch_size = batch_size
        # ratio of valid to invalid samples per batch for training ConvKB Model
        self.invalid_valid_ratio = int(valid_to_invalid_samples_ratio)

        self.graph_positive, self.graph_negative = self.get_graph()

        self.entity_in_dim = entity_dim
        self.relation_in_dim = relation_dim
        self.node_attention = torch.nn.MultiheadAttention(self.entity_in_dim, 1).cuda()
        self.realation_attention = torch.nn.MultiheadAttention(self.relation_in_dim, 1).cuda()

        if(get_2hop):
            self.node_neighbors_2hop_positive, self.node_neighbors_2hop_negative = self.get_further_neighbors()
        else:
            self.node_neighbors_2hop_positive = {}
            self.node_neighbors_2hop_negative = {}

        self.unique_entities_train = [self.entity2id[i] for i in unique_entities_train]

        self.train_indices = np.array(
            list(self.train_triples)).astype(np.int32)
        # These are valid triples, hence all have value 1
        self.train_values = np.array(
            [[1]] * len(self.train_triples)).astype(np.float32)

        self.validation_indices = np.array(
            list(self.validation_triples)).astype(np.int32)
        self.validation_values = np.array(
            [[1]] * len(self.validation_triples)).astype(np.float32)

        self.test_indices = np.array(list(self.test_triples)).astype(np.int32)
        self.test_values = np.array(
            [[1]] * len(self.test_triples)).astype(np.float32)

        self.valid_triples_dict = {j: i for i, j in enumerate(
            self.train_triples + self.validation_triples + self.test_triples)}
        logging.info("Nums of total triples: {}".format(len(self.valid_triples_dict)))
        logging.info("Nums of train triples: {}".format(len(self.train_indices)))
        logging.info("Nums of valid triples: {}".format(len(self.validation_indices)))
        logging.info("Nums of test triples: {}".format(len(self.test_indices)))

        # For training purpose
        self.batch_indices = np.empty(
            (self.batch_size * (self.invalid_valid_ratio + 1), 3)).astype(np.int32)
        self.batch_values = np.empty(
            (self.batch_size * (self.invalid_valid_ratio + 1), 1)).astype(np.float32)

    def get_further_neighbors(self, nbd_size=2):
        neighbors_positive = {}
        neighbors_negative = {}
        start_time = time.time()
        for index, source in enumerate(self.graph_positive.keys()):
            if index % 1000 == 0:
                logging.info("======positive dealed num:{}=====".format(index))
            # st_time = time.time()
            temp_neighbors = self.bfs(self.graph_positive, source, nbd_size)
            for distance in temp_neighbors.keys():
                if(source in neighbors_positive.keys()):
                    if(distance in neighbors_positive[source].keys()):
                        neighbors_positive[source][distance].append(
                            temp_neighbors[distance])
                    else:
                        neighbors_positive[source][distance] = temp_neighbors[distance]
                else:
                    neighbors_positive[source] = {}
                    neighbors_positive[source][distance] = temp_neighbors[distance]

        logging.info("===============positive graph created!==============")

        for index, source in enumerate(self.graph_negative.keys()):
            if index % 1000 == 0:
                logging.info("======negative dealed num:{}=====".format(index))
            # st_time = time.time()
            temp_neighbors = self.bfs(self.graph_negative, source, nbd_size)
            for distance in temp_neighbors.keys():
                if(source in neighbors_negative.keys()):
                    if(distance in neighbors_negative[source].keys()):
                        neighbors_negative[source][distance].append(
                            temp_neighbors[distance])
                    else:
                        neighbors_negative[source][distance] = temp_neighbors[distance]
                else:
                    neighbors_negative[source] = {}
                    neighbors_negative[source][distance] = temp_neighbors[distance]
        logging.info("===============negative graph created!==============")

        logging.info("Time taken: {}".format(time.time() - start_time))
        logging.info("Length of neighbors: {}".format(len(neighbors_positive)))
        return neighbors_positive, neighbors_negative

    def get_iteration_batch(self, iter_num):
        if (iter_num + 1) * self.batch_size <= len(self.train_indices):
            self.batch_indices = np.empty(
                (self.batch_size * (self.invalid_valid_ratio + 1), 3)).astype(np.int32)
            self.batch_values = np.empty(
                (self.batch_size * (self.invalid_valid_ratio + 1), 1)).astype(np.float32)

            indices = range(self.batch_size * iter_num,
                            self.batch_size * (iter_num + 1))

            self.batch_indices[:self.batch_size,
                               :] = self.train_indices[indices, :]
            self.batch_values[:self.batch_size,
                              :] = self.train_values[indices, :]

            last_index = self.batch_size

            if self.invalid_valid_ratio > 0:
                random_entities = np.random.randint(
                    0, len(self.entity2id), last_index * self.invalid_valid_ratio)

                # Precopying the same valid indices from 0 to batch_size to rest
                # of the indices
                self.batch_indices[last_index:(last_index * (self.invalid_valid_ratio + 1)), :] = np.tile(
                    self.batch_indices[:last_index, :], (self.invalid_valid_ratio, 1))
                self.batch_values[last_index:(last_index * (self.invalid_valid_ratio + 1)), :] = np.tile(
                    self.batch_values[:last_index, :], (self.invalid_valid_ratio, 1))

                for i in range(last_index):
                    for j in range(self.invalid_valid_ratio // 2):
                        current_index = i * (self.invalid_valid_ratio // 2) + j

                        while (random_entities[current_index], self.batch_indices[last_index + current_index, 1],
                               self.batch_indices[last_index + current_index, 2]) in self.valid_triples_dict.keys():
                            random_entities[current_index] = np.random.randint(
                                0, len(self.entity2id))
                        self.batch_indices[last_index + current_index,
                                           0] = random_entities[current_index]
                        self.batch_values[last_index + current_index, :] = [-1]

                    for j in range(self.invalid_valid_ratio // 2):
                        current_index = last_index * \
                            (self.invalid_valid_ratio // 2) + \
                            (i * (self.invalid_valid_ratio // 2) + j)

                        while (self.batch_indices[last_index + current_index, 0], self.batch_indices[last_index + current_index, 1],
                               random_entities[current_index]) in self.valid_triples_dict.keys():
                            random_entities[current_index] = np.random.randint(
                                0, len(self.entity2id))
                        self.batch_indices[last_index + current_index,
                                           2] = random_entities[current_index]
                        self.batch_values[last_index + current_index, :] = [-1]

                return self.batch_indices, self.batch_values

            return self.batch_indices, self.batch_values

        else:
            last_iter_size = len(self.train_indices) - \
                self.batch_size * iter_num
            self.batch_indices = np.empty(
                (last_iter_size * (self.invalid_valid_ratio + 1), 3)).astype(np.int32)
            self.batch_values = np.empty(
                (last_iter_size * (self.invalid_valid_ratio + 1), 1)).astype(np.float32)

            indices = range(self.batch_size * iter_num,
                            len(self.train_indices))
            self.batch_indices[:last_iter_size,
                               :] = self.train_indices[indices, :]
            self.batch_values[:last_iter_size,
                              :] = self.train_values[indices, :]

            last_index = last_iter_size

            if self.invalid_valid_ratio > 0:
                random_entities = np.random.randint(
                    0, len(self.entity2id), last_index * self.invalid_valid_ratio)

                # Precopying the same valid indices from 0 to batch_size to rest
                # of the indices
                self.batch_indices[last_index:(last_index * (self.invalid_valid_ratio + 1)), :] = np.tile(
                    self.batch_indices[:last_index, :], (self.invalid_valid_ratio, 1))
                self.batch_values[last_index:(last_index * (self.invalid_valid_ratio + 1)), :] = np.tile(
                    self.batch_values[:last_index, :], (self.invalid_valid_ratio, 1))

                for i in range(last_index):
                    for j in range(self.invalid_valid_ratio // 2):
                        current_index = i * (self.invalid_valid_ratio // 2) + j

                        while (random_entities[current_index], self.batch_indices[last_index + current_index, 1],
                               self.batch_indices[last_index + current_index, 2]) in self.valid_triples_dict.keys():
                            random_entities[current_index] = np.random.randint(
                                0, len(self.entity2id))
                        self.batch_indices[last_index + current_index,
                                           0] = random_entities[current_index]
                        self.batch_values[last_index + current_index, :] = [-1]

                    for j in range(self.invalid_valid_ratio // 2):
                        current_index = last_index * \
                            (self.invalid_valid_ratio // 2) + \
                            (i * (self.invalid_valid_ratio // 2) + j)

                        while (self.batch_indices[last_index + current_index, 0], self.batch_indices[last_index + current_index, 1],
                               random_entities[current_index]) in self.valid_triples_dict.keys():
                            random_entities[current_index] = np.random.randint(
                                0, len(self.entity2id))
                        self.batch_indices[last_index + current_index,
                                           2] = random_entities[current_index]
                        self.batch_values[last_index + current_index, :] = [-1]

                return self.batch_indices, self.batch_values

            return self.batch_indices, self.batch_values

    def get_nhop_neighbors_attention(self, batch_sources, node_neighbors,
                                                  entity_embeddings, relation_embeddings, nbd_size=2):
        batch_source_triples = []
        nodes_num = entity_embeddings.shape[0]
        relation_num = relation_embeddings.shape[0]
        count = 0
        for source in batch_sources:
            node_list = []
            node_list.append(source)
            if source in node_neighbors.keys():
                nhop_list = node_neighbors[source][nbd_size]
                for i, tup in enumerate(nhop_list):
                    node_list.append(tup[1][0])
                entity_embeddings_local = entity_embeddings[node_list, :]
                entity_q = torch.unsqueeze(entity_embeddings_local, 1)
                entity_k = torch.unsqueeze(entity_embeddings_local, 1)
                entity_v = torch.unsqueeze(entity_embeddings_local, 1)
                _, input_node_attention_weight = self.node_attention(entity_q, entity_k, entity_v)
                input_node_attention_weight = input_node_attention_weight.cpu().detach().numpy()
                input_node_attention_weight = input_node_attention_weight[:, 0, 1:]
                sorted_node_idx = self.get_topk_nodes(input_node_attention_weight, max_nodes=10)
                top_k_nodes = []
                for idx in sorted_node_idx:
                    top_k_nodes.append(node_list[1+idx])
                if len(top_k_nodes) >= 2:
                    sampled_nodes = np.random.choice(top_k_nodes, 2)
                else:
                    sampled_nodes = top_k_nodes
                for node_id in sampled_nodes:
                    second_relation_set = set([])
                    first_relation_id = -1
                    for i, tup in enumerate(nhop_list):
                        if node_id == tup[1][0]:
                            first_relation_id = tup[0][0]
                    for i, tup in enumerate(nhop_list):
                        if node_id == tup[1][0]:
                            second_relation_set.add(tup[0][1])

                    if first_relation_id != -1:
                        second_relation_list = [first_relation_id]
                        second_relation_list.extend(list(second_relation_set))
                        relation_embeddings_local = relation_embeddings[second_relation_list, :]
                        relation_q = torch.unsqueeze(relation_embeddings_local, 1)
                        relation_k = torch.unsqueeze(relation_embeddings_local, 1)
                        relation_v = torch.unsqueeze(relation_embeddings_local, 1)
                        _, input_relation_attention_weight = self.realation_attention(relation_q, relation_k, relation_v)
                        input_relation_attention_weight = input_relation_attention_weight.cpu().detach().numpy()
                        input_relation_attention_weight = input_relation_attention_weight[:, 0, 1:]
                        sorted_relation_idx = self.get_topk_nodes(input_relation_attention_weight, max_nodes=3)
                        top_k_relation = []
                        if len(sorted_relation_idx) >= 1:
                            for idx in sorted_relation_idx:
                                top_k_relation.append(second_relation_list[1+idx])
                            sampled_relations = np.random.choice(top_k_relation, 1)
                            for i, tup in enumerate(nhop_list):
                                if node_id == tup[1][0] and sampled_relations[0] == tup[0][1]:
                                    count += 1
                                    batch_source_triples.append([source, sampled_relations[0], first_relation_id,
                                                                 node_id])
        return np.array(batch_source_triples).astype(np.int32)

    def get_topk_nodes(self, node_attention, max_nodes):

        eps = 1e-20
        n_nodes = node_attention.shape[1]
        max_nodes = min(n_nodes, max_nodes)
        sorted_idx = np.squeeze(np.argsort(-node_attention, axis=1)[:, :max_nodes], 0)
        mask = node_attention[:, list(sorted_idx)] > eps
        mask = np.squeeze(mask, 0)
        real_sorted_idx = sorted_idx[mask]

        return real_sorted_idx

    def get_nhop_neighbors(self, batch_sources, node_neighbors, nbd_size=2):
        batch_source_triples = []
        count = 0
        for source in batch_sources:
            # randomly select from the list of neighbors
            if source in node_neighbors.keys():
                nhop_list = node_neighbors[source][nbd_size]

                for i, tup in enumerate(nhop_list):
                    if(config['partial_2hop'] and i >= 2):
                        break

                    count += 1
                    batch_source_triples.append([source, nhop_list[i][0][-1], nhop_list[i][0][0],
                                                 nhop_list[i][1][0]])

        return np.array(batch_source_triples).astype(np.int32)

    def get_iteration_batch_nhop(self, current_batch_indices, node_neighbors, batch_size):

        self.batch_indices = np.empty(
            (batch_size * (self.invalid_valid_ratio + 1), 4)).astype(np.int32)
        self.batch_values = np.empty(
            (batch_size * (self.invalid_valid_ratio + 1), 1)).astype(np.float32)
        indices = random.sample(range(len(current_batch_indices)), batch_size)

        self.batch_indices[:batch_size,
                           :] = current_batch_indices[indices, :]
        self.batch_values[:batch_size,
                          :] = np.ones((batch_size, 1))

        last_index = batch_size

        if self.invalid_valid_ratio > 0:
            random_entities = np.random.randint(
                0, len(self.entity2id), last_index * self.invalid_valid_ratio)

            # Precopying the same valid indices from 0 to batch_size to rest
            # of the indices
            self.batch_indices[last_index:(last_index * (self.invalid_valid_ratio + 1)), :] = np.tile(
                self.batch_indices[:last_index, :], (self.invalid_valid_ratio, 1))
            self.batch_values[last_index:(last_index * (self.invalid_valid_ratio + 1)), :] = np.tile(
                self.batch_values[:last_index, :], (self.invalid_valid_ratio, 1))

            for i in range(last_index):
                for j in range(self.invalid_valid_ratio // 2):
                    current_index = i * (self.invalid_valid_ratio // 2) + j

                    self.batch_indices[last_index + current_index,
                                       0] = random_entities[current_index]
                    self.batch_values[last_index + current_index, :] = [0]

                for j in range(self.invalid_valid_ratio // 2):
                    current_index = last_index * \
                        (self.invalid_valid_ratio // 2) + \
                        (i * (self.invalid_valid_ratio // 2) + j)

                    self.batch_indices[last_index + current_index,
                                       3] = random_entities[current_index]
                    self.batch_values[last_index + current_index, :] = [0]

            return self.batch_indices, self.batch_values

        return self.batch_indices, self.batch_values

    def get_graph(self):
        graph_positive = {}
        graph_negative = {}
        all_tiples = torch.cat([self.train_adj_matrix[0].transpose(
            0, 1), self.train_adj_matrix[1].unsqueeze(1)], dim=1)

        for data in all_tiples:
            source = data[1].data.item()
            target = data[0].data.item()
            value = data[2].data.item()

            if(source not in graph_positive.keys()):
                graph_positive[source] = {}
            if(target not in graph_negative.keys()):
                graph_negative[target] = {}
            graph_positive[source][target] = value
            graph_negative[target][source] = value
        return graph_positive, graph_negative

    def bfs(self, graph, source, nbd_size=2):
        visit = {}
        distance = {}
        parent = {}
        distance_lengths = {}

        visit[source] = 1
        distance[source] = 0
        parent[source] = (-1, -1)

        q = queue.Queue()
        q.put((source, -1))

        while(not q.empty()):
            top = q.get()
            if top[0] in graph.keys():
                for target in graph[top[0]].keys():
                    if(target in visit.keys()):
                        continue
                    else:
                        q.put((target, graph[top[0]][target]))

                        distance[target] = distance[top[0]] + 1

                        visit[target] = 1
                        if distance[target] > 2:
                            continue
                        parent[target] = (top[0], graph[top[0]][target])

                        if distance[target] not in distance_lengths.keys():
                            distance_lengths[distance[target]] = 1

        neighbors = {}
        for target in visit.keys():
            if(distance[target] != nbd_size):
                continue
            edges = [-1, parent[target][1]]
            relations = []
            entities = [target]
            temp = target
            while(parent[temp] != (-1, -1)):
                relations.append(parent[temp][1])
                entities.append(parent[temp][0])
                temp = parent[temp][0]

            if(distance[target] in neighbors.keys()):
                neighbors[distance[target]].append(
                    (tuple(relations), tuple(entities[:-1])))
            else:
                neighbors[distance[target]] = [
                    (tuple(relations), tuple(entities[:-1]))]

        return neighbors


    def transe_scoring(self, batch_inputs, entity_embeddings, relation_embeddings):
        source_embeds = entity_embeddings[batch_inputs[:, 0]]
        relation_embeds = relation_embeddings[batch_inputs[:, 1]]
        tail_embeds = entity_embeddings[batch_inputs[:, 2]]
        x = source_embeds + relation_embeds - tail_embeds
        x = torch.norm(x, p=1, dim=1)
        return x