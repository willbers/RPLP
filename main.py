from process.n_hop_data import load_data, load_nhop_data
from train_eval import train_gat, train_conv, evaluate
from config import RPGAT_config as config

def main():
    triples, entity_embeddings, relation_embeddings = load_data()
    nhop_nodes = load_nhop_data(triples)

    if config["train_gat"]:
        train_gat(triples, entity_embeddings, relation_embeddings, nhop_nodes)
    if config['train_conv']:
        train_conv(triples, entity_embeddings, relation_embeddings)
    if config['evaluate']:
        evaluate(triples, entity_embeddings, relation_embeddings, triples.unique_entities_train)

if __name__ == '__main__':
    main()