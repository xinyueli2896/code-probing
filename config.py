def config(task):
    layers = list((range(13)))
    size_dataset = 7000 if task == 'relational' else 10000
    hidden_size = 768
    