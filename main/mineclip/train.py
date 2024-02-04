import datetime
import json
from MCdata.data_loader import get_naive_dataloader
from mineclip.mineclip.optimizer import get_optimizer

from mineclip.utils.metrices import compute_metrices
from mineclip import MineCLIP
from mineclip.utils import get_args, set_seed, get_logger, save_state_dict
# get_optimizer, save_model, get_naive_dataloader, compute_metrics

import os
import time
import torch
from torch.amp import autocast
from tensorboardX import SummaryWriter


# Distributed Data Parallel: Check if we are in a distributed environment
if 'WORLD_SIZE' in os.environ:  # We are in a distributed environment
    local_rank = int(os.environ['LOCAL_RANK'])
    n_gpu = int(os.environ['WORLD_SIZE'])
else:   # We are in a single machine environment
    local_rank = 0
    n_gpu = 1
    os.environ['RANK'] = "0"
    os.environ['WORLD_SIZE'] = "1"
    os.environ['LOCAL_RANK'] = "0"
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

# Setup distributed training
torch.distributed.init_process_group(backend="nccl", timeout=datetime.timedelta(0, 18000))
_time = time.strftime("%y_%m_%d_%H:%M:%S", time.localtime())
assert local_rank == torch.distributed.get_rank(), \
    "local_rank {} is not equal to torch.distributed.get_rank() {}".format(local_rank, torch.distributed.get_rank())
assert n_gpu == torch.distributed.get_world_size(), \
    "n_gpu {} is not equal to torch.distributed.get_world_size() {}".format(n_gpu, torch.distributed.get_world_size())
print("local_rank: {}, n_gpu: {}".format(local_rank, n_gpu))

def contrastive_loss(logits_per_image, logits_per_text):
    """
    Calculate the loss of the model: contrastive loss
    https://zhuanlan.zhihu.com/p/624173920
    """
    labels = torch.arange(len(logits_per_image)).to(logits_per_image.device)
    image_loss = torch.nn.CrossEntropyLoss()(logits_per_image, labels)
    text_loss = torch.nn.CrossEntropyLoss()(logits_per_text, labels)
    total_loss = (image_loss + text_loss) / 2
    return total_loss



# clip_model = CLIP()


# clip_score_head = CLIPScoreHead(
#     clip_model,
#     video_adapter_layers=2,
#     text_adapter_layers=2,
#     feature_dim=512,
# )

# # 假设你有一些视频和文本特征
# video_feature = torch.randn(1, 512)  # 假设特征维度是 512
# texts = ["This is a sample text."]  # 假设你有一些文本

# # 使用 CLIPScoreHead 计算奖励分数
# logits_per_video, logits_per_text = clip_score_head(video_feature, texts)



def train_epoch(epoch, args, model, train_dataloader, device, optimizer, scheduler, global_step):
    """
    Train the model for one epoch.
    
    Args:
        epoch: current epoch number.
        args: arguments for training.
        model: the model to be trained.
        train_dataloader: the dataloader for training data.
        device: the device to be used for training.
        optimizer: the optimizer to be used for training.
        scheduler: the scheduler to be used for training.
        global_step: the current global step.
    """
    global logger

    model.train()   # Set the model to training mode
    log_step = args.n_display   # frequency of logging
    start_time = time.time()
    total_loss = 0  # total loss
    grad_step = 0   # number of gradient steps

    for step, batch in enumerate(train_dataloader):
        torch.cuda.empty_cache()
        video, text = tuple(t.to(device) for t in batch)
        # Preprocess text
        text_feats_batch = model.encode_text(text)

        with autocast(device_type='cuda'):  # Automatic Mixed Precision (AMP)
            logits_per_image, logits_per_text = model(video, text_tokens=text_feats_batch,  is_video_features=False, train=True)
            loss = contrastive_loss(logits_per_image, logits_per_text)

        loss.backward()

        total_loss += float(loss)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # Gradient clipping: avoid gradient explosion

        if scheduler is not None:   # learning rate update
            scheduler.step()

        optimizer.step()
        optimizer.zero_grad()

        model.module.clamp_logit_scale()    # Clamp the logit scale

        global_step += 1
        grad_step += 1
        if global_step % log_step == 0 and local_rank == 0:
            logger.info("Epoch: %d/%s, Step: %d/%d, Lr: %s, Loss: %f, Time/step: %f", epoch + 1,
                        args.epochs, step + 1,
                        len(train_dataloader),
                        "-".join([str('%.9f' % itm) for itm in sorted(list(set(optimizer.get_lr())))]),
                        float(loss),
                        (time.time() - start_time) / log_step)
            start_time = time.time()

    total_loss = total_loss / grad_step # average loss

    return total_loss, global_step


def eval_epoch(model, test_dataloader, writer, epoch, device):
    """
    Evaluate the model on the test dataset.

    Args:
        model: the model to be evaluated.
        test_dataloader: the dataloader for test data.
        writer: the writer to write the evaluation results.
        epoch: current epoch number.
        device: the device to be used for evaluation.
    """

    model.eval()

    batch_list_t = []   # list of text features
    batch_list_v = []   # list of video features
    with torch.no_grad():

        for bid, batch in enumerate(test_dataloader):
            torch.cuda.empty_cache()
            batch = tuple(t.to(device) for t in batch)
            with autocast(device_type='cuda'):
                video_features, text_features = model(*batch, train=False)
            
            if local_rank == 0:
                if isinstance(video_features, list):
                    if len(batch_list_v) == 0:
                        batch_list_v = [[] for _ in range(len(video_features))]
                    for i in range(len(video_features)):
                        batch_list_v[i].append(video_features[i].cpu())
                else:
                    batch_list_v.append(video_features.cpu())

                if isinstance(text_features, list):
                    if len(batch_list_t) == 0:
                        batch_list_t = [[] for _ in range(len(text_features))]
                    for i in range(len(text_features)):
                        batch_list_t[i].append(text_features[i].cpu())
                else:
                    batch_list_t.append(text_features.cpu())

            print("{}/{}\r".format(bid, len(test_dataloader)), end="")

        # Concatenate the features into a single tensor
        if local_rank == 0:
            if isinstance(batch_list_v[0], list):   # multiple kinds of video features
                kind = len(batch_list_v)
                video_features = [torch.cat(itm, dim=0) for itm in batch_list_v]
            else:
                kind = 1
                video_features = torch.cat(batch_list_v, dim=0)
                
            if isinstance(batch_list_t[0], list):   # multiple kinds of text features
                if kind == 1:
                    kind = len(batch_list_t)
                else:
                    assert kind == len(batch_list_t)
                text_features = [torch.cat(itm, dim=0) for itm in batch_list_t]
            else:
                text_features = torch.cat(batch_list_t, dim=0)

            final_sim_matrix = 0

            for ki in range(kind):
                sub_video_features = video_features[ki] if isinstance(video_features, list) else video_features
                sub_text_features = text_features[ki] if isinstance(text_features, list) else text_features
                sim_matrix = sub_video_features @ sub_text_features.t()
                final_sim_matrix += sim_matrix

            # Compute the metrics for the final similarity matrix
            vt_metrics = compute_metrices(final_sim_matrix.cpu().numpy())
            tv_metrics = compute_metrices(final_sim_matrix.cpu().numpy().T)

            # Write the metrics to the tensorboard
            logger.info("Video-to-Text:")
            logger.info('\t>>>  V2T$R@1: {:.1f} - V2T$R@5: {:.1f} - V2T$R@10: {:.1f}'
                        ' - V2T$Median R: {:.1f} - V2T$Mean R: {:.1f}'.
                        format(vt_metrics['R1'], vt_metrics['R5'], vt_metrics['R10'],
                               vt_metrics['MedianR'], vt_metrics['MeanR']))
            logger.info("Text-to-Video:")
            logger.info('\t>>>  T2V$R@1: {:.1f} - T2V$R@5: {:.1f} - T2V$R@10: {:.1f}'
                        ' - T2V$Median R: {:.1f} - T2V$Mean R: {:.1f}'.
                        format(tv_metrics['R1'], tv_metrics['R5'], tv_metrics['R10'],
                               tv_metrics['MedianR'], tv_metrics['MeanR']))

            for k, v in tv_metrics.items():
                writer.add_scalar("V2T_{}/{}".format('all', k), v, epoch)
            for k, v in vt_metrics.items():
                writer.add_scalar("T2V_{}/{}".format('all', k), v, epoch)


def main(args):
    global logger

    print("Prepare training...")
    # Setup output directory './ckpt'
    output_dir = os.path.join(args.save_model_path, _time)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    # Save the logger into a file in the output directory
    log_file_name = os.path.join(output_dir, 'training_log.txt')
    logger = get_logger(log_file_name)

    # Setup CUDA, GPU & distributed training
    set_seed(args.seed)
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    logger.info("device: {} n_gpu: {}".format(device, n_gpu))

    # Setup logging
    if local_rank == 0:
        logger.info("Effective parameters:")
        for key in sorted(args.__dict__):
            logger.info("\t{}: {}".format(key, args.__dict__[key]))

        with open(os.path.join(output_dir, 'args.json'), 'w') as f:
            json.dump(args.__dict__, f, indent=2)
        os.system('cp {} {}'.format('config.yaml', output_dir))

        writer_dir = os.path.join(output_dir, 'runs')
        logger.info("writer_dir: {}".format(writer_dir))
        train_writer = SummaryWriter(os.path.join(writer_dir, 'train'))
        test_writer = SummaryWriter(os.path.join(writer_dir, 'test'))
    else:
        train_writer = None
        test_writer = None

    # Setup Model
    if args.model_type == 'MineCLIP':
        model = MineCLIP(arch="vit_base_p16_fz.v2.t2", 
            resolution=[160, 256], 
            pool_type='attn', 
            image_feature_dim=512, 
            mlp_adapter_spec='v0-2.t0', 
            hidden_dim=512)
    else:
        raise NotImplementedError

    if args.use_pretrained_model:
        if local_rank == 0:
            logger.info("Loading pretrained model from {}".format(args.pretrain_model_path))
        model.load_state_dict(torch.load(args.pretrain_model_path), strict=False)

    model = model.to(device)

    # Setup training dataset
    train_dataloader, train_sampler, train_length \
        = get_naive_dataloader(args.train_dataset_log_file, args.use_mask, args.batch_size, 'train', args.num_workers)
    # Setup test dataset
    test_dataloader, test_sampler, test_length \
        = get_naive_dataloader(args.test_dataset_log_file, args.use_mask, args.batch_size_eval, 'test', args.num_workers)

    num_train_optimization_steps = train_length // args.batch_size * args.epochs

    if local_rank == 0:
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", train_length)
        logger.info("  Batch size = %d", args.batch_size)
        logger.info("  Num steps = %d", num_train_optimization_steps)
        logger.info("***** Running test *****")
        logger.info("  Num examples = %d", test_length)
        logger.info("  Batch size = %d", args.batch_size_eval)
        logger.info("  Num steps = %d", len(test_dataloader))

    # Prepare optimizer
    optimizer = get_optimizer(optimizer_name=args.optimizer_name,
                              schedule_name=args.schedule_name,
                              model=model,
                              lr=args.lr,
                              layer_wise_lr_decay=args.layer_wise_lr_decay,
                              weight_decay=args.weight_decay,
                              warmup_proportion=args.warmup_proportion,
                              t_total=num_train_optimization_steps,
                              max_grad_norm=args.max_grad_norm,
                              text_freeze_layer=args.text_freeze_layer,
                              video_freeze_layer=args.video_freeze_layer)

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank],
                                                      output_device=local_rank, find_unused_parameters=True)

    scheduler = None

    print("Start training...")
    global_step = 0
    for epoch in range(args.epochs):
        print("------The {} round--------------".format(epoch+1))
        train_sampler.set_epoch(epoch)
        test_sampler.set_epoch(epoch)
        tr_loss, global_step = train_epoch(epoch, args, model, train_dataloader, device, optimizer,
                                           scheduler, global_step)
        if local_rank == 0:
            logger.info("Epoch %d/%s Finished, Train Loss: %f", epoch + 1, args.epochs, tr_loss)
            train_writer.add_scalar('loss', tr_loss, epoch + 1)
            save_state_dict(obj=model, model_path=args.save_model_path, serial=1, type_name="")

        if local_rank == 0:
            logger.info("Eval on test dataset")
        eval_epoch(model, test_dataloader, test_writer, epoch, device)


if __name__ == "__main__":
    main(get_args())
