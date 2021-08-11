import torch

from config import RPGAT_config as config

def batch_loss(gat_loss_func, train_indices, relation_embed, entity_relation_embed):
    len_pos_triples = int(train_indices.shape[0] / (int(config['valid_invalid_ratio_gat']) + 1))

    pos_triples = train_indices[:len_pos_triples]
    neg_triples = train_indices[len_pos_triples:]

    pos_triples = pos_triples.repeat(int(config['valid_invalid_ratio_gat']), 1)

    source_embeds = entity_relation_embed[pos_triples[:, 1], pos_triples[:, 0]]
    relation_embeds = relation_embed[pos_triples[:, 1]]
    tail_embeds = entity_relation_embed[pos_triples[:, 1], pos_triples[:, 2]]

    x = source_embeds + relation_embeds - tail_embeds
    pos_norm = torch.norm(x, p=1, dim=1)

    source_embeds = entity_relation_embed[neg_triples[:, 1], neg_triples[:, 0]]
    relation_embeds = relation_embed[neg_triples[:, 1]]
    tail_embeds = entity_relation_embed[neg_triples[:, 1], neg_triples[:, 2]]

    x = source_embeds + relation_embeds - tail_embeds
    neg_norm = torch.norm(x, p=1, dim=1)

    y = -torch.ones(int(config['valid_invalid_ratio_gat']) * len_pos_triples).cuda()

    loss = gat_loss_func(pos_norm, neg_norm, y)
    return loss