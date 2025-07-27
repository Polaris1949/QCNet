import argparse
import os

DATASET_ROOT = './data_av2'
CHECKPOINT = './lightning_logs/version_73/checkpoints/epoch=49-step=20000.ckpt'
is_load_ckpt = True  # 是否加载预训练模型
ckpt_dir = r'/home/lk/QCNet/lightning_logs/version_73/checkpoints/epoch=49-step=20000.ckpt'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', nargs='?', type=str, choices=['train', 'val', 'test'], default='train', help='Mode to run the script in')
    args = parser.parse_args()
    mode = args.mode

    print(f"Running in {mode} mode")

    if mode == 'train':
        os.system(
            'python train_qcnet.py '
            f'--root {DATASET_ROOT} '
            '--train_batch_size 1 '
            '--val_batch_size 1 '
            '--test_batch_size 1 '
            '--devices 1 '
            '--dataset argoverse_v2 '
            '--num_historical_steps 50 '
            '--num_future_steps 60 '
            '--num_recurrent_steps 3 '
            '--pl2pl_radius 150 '
            '--time_span 10 '
            '--pl2a_radius 50 '
            '--a2a_radius 50 '
            '--num_t2m_steps 30 '
            '--pl2m_radius 150 '
            '--a2m_radius 50 '  # the origin data is 150m
            '--hidden_dim 128 ' # TODO
            '--max_epochs 150 '
            '--load_ckpt' + f' {str(is_load_ckpt)} '
            '--ckpt_dir ' + ckpt_dir + ' '
        )
    elif mode == 'val':
        os.system(
            'python val.py '
            '--model QCNet '
            f'--root {DATASET_ROOT} '
            f'--ckpt_path {CHECKPOINT}'
        )
    elif mode == 'test':
        os.system(
            'python test.py '
            '--model QCNet '
            '--batch_size 1 '
            f'--root {DATASET_ROOT} '
            f'--ckpt_path {CHECKPOINT}'
        )
    else:
        raise ValueError("Invalid mode selected. Choose from 'train', 'val', or 'test'.")

    print("Script execution completed.")
