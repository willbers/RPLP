# params for our model RPGAT

RPGAT_config = {
    "train_gat": True,
    "train_conv": True,
    "evaluate": True,


    'data': "./data/FB15k-237/",
    'output_folder': './checkpoints/fb/out/',
    'get_2hop': False,
    'use_2hop': True,
    'partial_2hop': True,

    'epochs_gat': 20,
    'epochs_conv': 20,
    'batch_size_gat': 280000,
    'batch_size_conv': 128,
    'weight_decay_gat':1e-5,
    'weight_decay_conv': 1e-6,

    'pretrained_emb': True,
    'embedding_size': 50,
    'entity_out_dim': [100, 200],
    'relation_out_dim': [100, 200],
    'conv_out_channels': 50,
    'lr': 1e-3,
    # Ratio of valid to invalid triples for GAT training
    'valid_invalid_ratio_gat': 2,
    'drop_GAT': 0.3,
    # LeakyRelu alphs in GAT
    'alpha': 0.2,
    # Multihead attention
    'nheads_GAT': [2, 2],
    # Margin in hingle-loss
    'margin': 1,
    # LeakyRelu alphs in Conv
    'alpha_conv': 0.2,
    # Ratio of valid to invalid triples for Conv training
    'valid_invalid_ratio_conv': 40,
    # Number of output channels in conv layer
    'drop_conv':0.3
}