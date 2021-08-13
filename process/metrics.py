import time
import numpy as np
import torch
from config import RPLP_config as config
import logging
logging.basicConfig(format='%(asctime)s - %(levelname)s: %(message)s',
                    level=logging.INFO)

def validation(triples, model, unique_entities):
    average_hits_at_100_head, average_hits_at_100_tail = [], []
    average_hits_at_ten_head, average_hits_at_ten_tail = [], []
    average_hits_at_three_head, average_hits_at_three_tail = [], []
    average_hits_at_one_head, average_hits_at_one_tail = [], []
    average_mean_rank_head, average_mean_rank_tail = [], []
    average_mean_recip_rank_head, average_mean_recip_rank_tail = [], []


    indices = [i for i in range(len(triples.test_indices))]
    batch_indices = triples.test_indices[indices, :]

    logging.info("Nums of samples in testing: {}".format(len(indices)))
    entity_list = [j for i, j in triples.entity2id.items()]

    ranks_head, ranks_tail = [], []
    reciprocal_ranks_head, reciprocal_ranks_tail = [], []
    hits_at_100_head, hits_at_100_tail = 0, 0
    hits_at_ten_head, hits_at_ten_tail = 0, 0
    hits_at_three_head, hits_at_three_tail = 0, 0
    hits_at_one_head, hits_at_one_tail = 0, 0

    for i in range(batch_indices.shape[0]):
        new_x_batch_head = np.tile(
            batch_indices[i, :], (len(triples.entity2id), 1))
        new_x_batch_tail = np.tile(
            batch_indices[i, :], (len(triples.entity2id), 1))

        if (batch_indices[i, 0] not in unique_entities or batch_indices[i, 2] not in unique_entities):
            continue

        new_x_batch_head[:, 0] = entity_list
        new_x_batch_tail[:, 2] = entity_list

        last_index_head = []  # array of already existing triples
        last_index_tail = []
        for tmp_index in range(len(new_x_batch_head)):
            temp_triple_head = (new_x_batch_head[tmp_index][0], new_x_batch_head[tmp_index][1],
                                new_x_batch_head[tmp_index][2])
            if temp_triple_head in triples.valid_triples_dict.keys():
                last_index_head.append(tmp_index)

            temp_triple_tail = (new_x_batch_tail[tmp_index][0], new_x_batch_tail[tmp_index][1],
                                new_x_batch_tail[tmp_index][2])
            if temp_triple_tail in triples.valid_triples_dict.keys():
                last_index_tail.append(tmp_index)

        new_x_batch_head = np.delete(
            new_x_batch_head, last_index_head, axis=0)
        new_x_batch_tail = np.delete(
            new_x_batch_tail, last_index_tail, axis=0)

        # adding the current valid triples to the top, i.e, index 0
        new_x_batch_head = np.insert(
            new_x_batch_head, 0, batch_indices[i], axis=0)
        new_x_batch_tail = np.insert(
            new_x_batch_tail, 0, batch_indices[i], axis=0)

        import math
        # Have to do this, because it doesn't fit in memory

        if 'WN' in config['data']:
            num_triples_each_shot = int(
                math.ceil(new_x_batch_head.shape[0] / 4))

            scores1_head = model.batch_test(torch.LongTensor(
                new_x_batch_head[:num_triples_each_shot, :]).cuda())
            scores2_head = model.batch_test(torch.LongTensor(
                new_x_batch_head[num_triples_each_shot: 2 * num_triples_each_shot, :]).cuda())
            scores3_head = model.batch_test(torch.LongTensor(
                new_x_batch_head[2 * num_triples_each_shot: 3 * num_triples_each_shot, :]).cuda())
            scores4_head = model.batch_test(torch.LongTensor(
                new_x_batch_head[3 * num_triples_each_shot: 4 * num_triples_each_shot, :]).cuda())

            scores_head = torch.cat(
                [scores1_head, scores2_head, scores3_head, scores4_head], dim=0)
        else:
            scores_head = model.batch_test(new_x_batch_head)

        sorted_scores_head, sorted_indices_head = torch.sort(
            scores_head.view(-1), dim=-1, descending=True)
        # Just search for zeroth index in the sorted scores, we appended valid triple at top
        ranks_head.append(
            np.where(sorted_indices_head.cpu().numpy() == 0)[0][0] + 1)
        reciprocal_ranks_head.append(1.0 / ranks_head[-1])

        # Tail part here

        if 'WN' in config['data']:
            num_triples_each_shot = int(
                math.ceil(new_x_batch_tail.shape[0] / 4))

            scores1_tail = model.batch_test(torch.LongTensor(
                new_x_batch_tail[:num_triples_each_shot, :]).cuda())
            scores2_tail = model.batch_test(torch.LongTensor(
                new_x_batch_tail[num_triples_each_shot: 2 * num_triples_each_shot, :]).cuda())
            scores3_tail = model.batch_test(torch.LongTensor(
                new_x_batch_tail[2 * num_triples_each_shot: 3 * num_triples_each_shot, :]).cuda())
            scores4_tail = model.batch_test(torch.LongTensor(
                new_x_batch_tail[3 * num_triples_each_shot: 4 * num_triples_each_shot, :]).cuda())

            scores_tail = torch.cat(
                [scores1_tail, scores2_tail, scores3_tail, scores4_tail], dim=0)

        else:
            scores_tail = model.batch_test(new_x_batch_tail)

        sorted_scores_tail, sorted_indices_tail = torch.sort(
            scores_tail.view(-1), dim=-1, descending=True)

        # Just search for zeroth index in the sorted scores, we appended valid triple at top
        ranks_tail.append(
            np.where(sorted_indices_tail.cpu().numpy() == 0)[0][0] + 1)
        reciprocal_ranks_tail.append(1.0 / ranks_tail[-1])
        if i % 1000 == 0:
            logging.info("Predicted num: {}".format(i))

    for i in range(len(ranks_head)):
        if ranks_head[i] <= 100:
            hits_at_100_head = hits_at_100_head + 1
        if ranks_head[i] <= 10:
            hits_at_ten_head = hits_at_ten_head + 1
        if ranks_head[i] <= 3:
            hits_at_three_head = hits_at_three_head + 1
        if ranks_head[i] == 1:
            hits_at_one_head = hits_at_one_head + 1

    for i in range(len(ranks_tail)):
        if ranks_tail[i] <= 100:
            hits_at_100_tail = hits_at_100_tail + 1
        if ranks_tail[i] <= 10:
            hits_at_ten_tail = hits_at_ten_tail + 1
        if ranks_tail[i] <= 3:
            hits_at_three_tail = hits_at_three_tail + 1
        if ranks_tail[i] == 1:
            hits_at_one_tail = hits_at_one_tail + 1

    assert len(ranks_head) == len(reciprocal_ranks_head)
    assert len(ranks_tail) == len(reciprocal_ranks_tail)

    logging.info("=" * 15 + "Stats for replacing head" + "=" * 15)
    logging.info("Hits@100 are {}".format(hits_at_100_head / float(len(ranks_head))))
    logging.info("Hits@10 are {}".format(hits_at_ten_head / len(ranks_head)))
    logging.info("Hits@3 are {}".format(hits_at_three_head / len(ranks_head)))
    logging.info("Hits@1 are {}".format(hits_at_one_head / len(ranks_head)))
    logging.info("Mean rank {}".format(sum(ranks_head) / len(ranks_head)))
    logging.info("Mean Reciprocal Rank {}".format(sum(reciprocal_ranks_head) / len(reciprocal_ranks_head)))

    logging.info("=" * 15 + "Stats for replacing tail" + "=" * 15)
    logging.info("Hits@100 are {}".format(hits_at_100_tail / len(ranks_head)))
    logging.info("Hits@10 are {}".format(hits_at_ten_tail / len(ranks_head)))
    logging.info("Hits@3 are {}".format(hits_at_three_tail / len(ranks_head)))
    logging.info("Hits@1 are {}".format(hits_at_one_tail / len(ranks_head)))
    logging.info("Mean rank {}".format(sum(ranks_tail) / len(ranks_tail)))
    logging.info("Mean Reciprocal Rank {}".format(sum(reciprocal_ranks_tail) / len(reciprocal_ranks_tail)))

    average_hits_at_100_head.append(hits_at_100_head / len(ranks_head))
    average_hits_at_ten_head.append(hits_at_ten_head / len(ranks_head))
    average_hits_at_three_head.append(hits_at_three_head / len(ranks_head))
    average_hits_at_one_head.append(hits_at_one_head / len(ranks_head))
    average_mean_rank_head.append(sum(ranks_head) / len(ranks_head))
    average_mean_recip_rank_head.append(sum(reciprocal_ranks_head) / len(reciprocal_ranks_head))

    average_hits_at_100_tail.append(hits_at_100_tail / len(ranks_head))
    average_hits_at_ten_tail.append(hits_at_ten_tail / len(ranks_head))
    average_hits_at_three_tail.append(hits_at_three_tail / len(ranks_head))
    average_hits_at_one_tail.append(hits_at_one_tail / len(ranks_head))
    average_mean_rank_tail.append(sum(ranks_tail) / len(ranks_tail))
    average_mean_recip_rank_tail.append(sum(reciprocal_ranks_tail) / len(reciprocal_ranks_tail))

    cumulative_hits_100 = (sum(average_hits_at_100_head) / len(average_hits_at_100_head)
                           + sum(average_hits_at_100_tail) / len(average_hits_at_100_tail)) / 2
    cumulative_hits_ten = (sum(average_hits_at_ten_head) / len(average_hits_at_ten_head)
                           + sum(average_hits_at_ten_tail) / len(average_hits_at_ten_tail)) / 2
    cumulative_hits_three = (sum(average_hits_at_three_head) / len(average_hits_at_three_head)
                             + sum(average_hits_at_three_tail) / len(average_hits_at_three_tail)) / 2
    cumulative_hits_one = (sum(average_hits_at_one_head) / len(average_hits_at_one_head)
                           + sum(average_hits_at_one_tail) / len(average_hits_at_one_tail)) / 2
    cumulative_mean_rank = (sum(average_mean_rank_head) / len(average_mean_rank_head)
                            + sum(average_mean_rank_tail) / len(average_mean_rank_tail)) / 2
    cumulative_mean_recip_rank = (sum(average_mean_recip_rank_head) / len(average_mean_recip_rank_head) + sum(
        average_mean_recip_rank_tail) / len(average_mean_recip_rank_tail)) / 2

    logging.info("=" * 15 + "Cumulative stats" + "=" * 15)
    logging.info("Hits@100 are {}".format(cumulative_hits_100))
    logging.info("Hits@10 are {}".format(cumulative_hits_ten))
    logging.info("Hits@3 are {}".format(cumulative_hits_three))
    logging.info("Hits@1 are {}".format(cumulative_hits_one))
    logging.info("Mean rank {}".format(cumulative_mean_rank))
    logging.info("Mean Reciprocal Rank {}".format(cumulative_mean_recip_rank))
