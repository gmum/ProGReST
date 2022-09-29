import argparse


def get_args():
    parser = argparse.ArgumentParser(description='ProtoGraphTree')
    parser.add_argument('--pt', type=float, default=0)
    parser.add_argument('--cl', type=float, default=0)
    parser.add_argument('--ld', type=float, default=0)


    parser.add_argument('--lr', type=float, default=0.05)
    parser.add_argument('--batch_size', type=int, default=48)
    parser.add_argument('--depth', type=int, default=5)
    parser.add_argument('--classes', type=int, default=1)
    parser.add_argument('--distance', type=str, default='euclidean')
    parser.add_argument('--epoch', type=int, default=250)


    parser.add_argument('--task_type', type=str, default='reg')
    parser.add_argument('--metric', type=str, default='acc')

    parser.add_argument('--dataset_name', type=str, default='Caco2_Wang')
    parser.add_argument('--split_method', type=str, default='default')
    parser.add_argument('--cache_encodings', type=bool, default=True)
    parser.add_argument('--split_seed', type=int, default=1)

    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--latent_distance', type=str, default='parents')

    parser.add_argument('--prototype_size', type=int, default=128)
    parser.add_argument('--features_size', type=int, default=768)

    parser.add_argument('--warmup', type=int, default=25)
    parser.add_argument('--join', type=int, default=100)

    parser.add_argument('--project_start', type=int, default=20)
    parser.add_argument('--project_mod', type=int, default=10)

    return parser.parse_args()
