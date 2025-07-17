import argparse
import os

IS_IN_EDITOR = True
DATASET_ROOT = './data_av2'
CHECKPOINT = './checkpoints/qcnet.ckpt'

if __name__ == '__main__':
    if IS_IN_EDITOR:
        mode = 'train'
    else:
        parser = argparse.ArgumentParser()
        parser.add_argument('mode', type=str, choices=['train', 'val', 'test'], help='Mode to run the script in')
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
            '--a2m_radius 150 '
            '--hidden_dim 1 ' # TODO
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
            f'--root {DATASET_ROOT} '
            f'--ckpt_path {CHECKPOINT}'
        )
    else:
        raise ValueError("Invalid mode selected. Choose from 'train', 'val', or 'test'.")

    print("Script execution completed.")
