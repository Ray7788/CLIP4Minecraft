import argparse

def get_args(description='MineCLIP args'):
    """
    Get arguments for MineCLIP
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--seed', type=int, default=42, help='set random seed') # 260817
    parser.add_argument('--n_display', type=int, default=10, help='display step')

    parser.add_argument('--train_dataset_log_file', type=str, default='./MCdata/train_videos_2/train_2.json',
                        help='training dataset log file of data(video&text input) paths')
    parser.add_argument('--test_dataset_log_file', type=str, default='./MCdata/test_videos_2/test_2.json',
                        help='test dataset log file of data(video&text input) paths')
    
    parser.add_argument('--use_pretrained_CLIP', action='store_true', default=False, help='use pretrained CLIP(ViT) model')
    parser.add_argument('--pretrained_CLIP_path', type=str, default="./ViT-B-16.pt", help='pretrained CLIP model path')

    parser.add_argument('--use_pretrained_model', action='store_true', default=False, help='use pretrained model')
    parser.add_argument('--pretrain_model_path', type=str, default="./CLIP4MC.pt", help='pretrained model path')
    parser.add_argument('--model_type', type=str, default='MineCLIP', choices=['CLIP4MC', 'MineCLIP'], help='pretrained model type')

    parser.add_argument('--clip_frame_num', type=int, default=16, help='frame num for each shorter clip')
    parser.add_argument('--clip_frame_stride', type=int, default=8, help='frame stride for each shorter clip')
    parser.add_argument('--use_mask', action='store_true', default=False, help='data process name')
    parser.add_argument('--num_workers', type=int, default=8, help='num workers')
    
    parser.add_argument('--batch_size', type=int, default=60, help='batch size for training')
    parser.add_argument('--batch_size_eval', type=int, default=60, help='batch size for evaluation')

    # Training
    parser.add_argument('--epochs', type=int, default=40, help='num of epochs')
    parser.add_argument('--optimizer_name', type=str, default="BertAdam", help='optimizer name')
    parser.add_argument('--schedule_name', type=str, default="warmup_cosine", help='schedule name')
    parser.add_argument('--lr', type=float, default=1.5e-4, help='initial learning rate')
    parser.add_argument('--layer_wise_lr_decay', type=float, default=0.65, help='coefficient for bert branch.')
    parser.add_argument('--weight_decay', type=float, default=0.2, help='Learning rate exp epoch decay')
    parser.add_argument("--warmup_proportion", default=0.005, type=float, help="Warmup proportion")
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help='max grad norm')

    # Freeze layer
    parser.add_argument('--text_freeze_layer', type=int, default=11, help='text encoder freeze layer')
    parser.add_argument('--video_freeze_layer', type=int, default=11, help='video encoder freeze layer')

    parser.add_argument('--save_model_path', type=str, default="./ckt", help='path for saving new trained model')

    args = parser.parse_args()
    local_rank = 0  # if not using distributed training
    args.seed = args.seed + local_rank

    return args