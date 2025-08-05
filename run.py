import argparse
import os

DATASET_ROOT = './data_av2'
CHECKPOINT = './lightning_logs/version_73/checkpoints/epoch=49-step=20000.ckpt'
IS_LOAD_CKPT = False  # 是否加载预训练模型
USE_NATSUMI = True  # 是否使用Natsumi模型
IS_LOAD_NATSUMI_CKPT = False  # 是否加载Natsumi预训练模型
NATSUMI_CKPT = './pretrained/natsumi.ckpt'  # Natsumi预训练模型的路径
NATSUMI_FREEZE = True  # 是否冻结Natsumi模型的参数
NATSUMI_FEAT_QCNET = True  # 是否使用QCNet的特征作为Natsumi输入
NUM_GRLC_STEPS = 10  # GRLC步骤数
SAVE_GRLC_STRUCTURE = True  # 是否保存GRLC结构

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', nargs='?', type=str, choices=['train', 'val', 'test'], default='train', help='Mode to run the script in')
    args = parser.parse_args()
    mode = args.mode

    print(f"Running in {mode} mode")

    if mode == 'train':
        ckpt_option = f'--ckpt_path {CHECKPOINT}' if IS_LOAD_CKPT else ''
        natsumi_option = f'--natsumi' if USE_NATSUMI else ''
        natsumi_ckpt_option = f'--natsumi_ckpt {NATSUMI_CKPT}' if IS_LOAD_NATSUMI_CKPT else ''
        natsumi_freeze_option = '--natsumi_freeze' if NATSUMI_FREEZE else ''
        natsumi_feat_qcnet_option = '--natsumi_feat_qcnet' if NATSUMI_FEAT_QCNET else ''
        save_grlc_struct_option = '--save_grlc_structure' if SAVE_GRLC_STRUCTURE else ''
        cmd = (
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
            f'{ckpt_option} '
            f'{natsumi_option} '  # 是否使用Natsumi模型
            f'{natsumi_ckpt_option} '  # Natsumi预训练模型的路径
            f'{natsumi_freeze_option} '  # 是否冻结Natsumi模型的参数
            f'{natsumi_feat_qcnet_option} '  # 是否使用QCNet的特征作为Natsumi输入
            f'--num_grlc_steps {NUM_GRLC_STEPS} '  # GRLC步骤数
            f'{save_grlc_struct_option} '  # 是否保存GRLC结构
        )
    elif mode == 'val':
        cmd = (
            'python val.py '
            '--model QCNet '
            f'--root {DATASET_ROOT} '
            f'--ckpt_path {CHECKPOINT}'
        )
    elif mode == 'test':
        cmd = (
            'python test.py '
            '--model QCNet '
            '--batch_size 1 '
            f'--root {DATASET_ROOT} '
            f'--ckpt_path {CHECKPOINT}'
        )
    else:
        raise ValueError("Invalid mode selected. Choose from 'train', 'val', or 'test'.")

    print(f"Executing command: {cmd}")
    ec = os.system(cmd)
    print(f"Command exit code: {ec}")
