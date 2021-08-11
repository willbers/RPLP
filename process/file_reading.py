def data_loading(filename, entity2id, relation2id, is_unweigted=False, directed=True):
    with open(filename) as f:
        lines = f.readlines()

    triples_data = []
    rows, cols, data = [], [], []
    unique_entities = set()

    for index, line in enumerate(lines):
        line = line.strip().split()
        e1 = line[0].strip()
        relation = line[1].strip()
        e2 = line[2].strip()
        unique_entities.add(e1)
        unique_entities.add(e2)
        triples_data.append((entity2id[e1], relation2id[relation], entity2id[e2]))
        if not directed:
            rows.append(entity2id[e1])
            cols.append(entity2id[e2])
            if is_unweigted:
                data.append(1)
            else:
                data.append(relation2id[relation])

        rows.append(entity2id[e2])
        cols.append(entity2id[e1])
        if is_unweigted:
            data.append(1)
        else:
            data.append(relation2id[relation])

    return triples_data, (rows, cols, data), list(unique_entities)


def entity_reading(filename):
    entity2id = {}
    with open(filename, 'r') as f:
        for line in f:
            if len(line.strip().split()) > 1:
                line = line.strip().split()
                entity = line[0].strip()
                entity_id = line[1].strip()
                entity2id[entity] = int(entity_id)
    return entity2id


def relation_reading(filename):
    relation2id = {}
    with open(filename, 'r') as f:
        for line in f:
            if len(line.strip().split()) > 1:
                line = line.strip().split()
                relation = line[0].strip()
                relation_id = line[1].strip()
                relation2id[relation] = int(relation_id)
    return relation2id