import torch
#from torch.utils.tensorboard import SummaryWriter
import argparse
import yaml
import os
import os.path as osp
from utils import get_logger
from model import build_model
from datasets import build_data_loader
from criterions import build_criterion
from trainer import build_trainer, build_eval
from utils import *
    
import torch.utils.benchmark as benchmark

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/miniroad_assembly.yaml')
    parser.add_argument('--eval', type=str, default=None)
    parser.add_argument('--amp', action='store_true')
    parser.add_argument('--tensorboard', action='store_true')
    parser.add_argument('--lr_scheduler', action='store_true')
    parser.add_argument('--no_rgb', action='store_true')
    parser.add_argument('--no_flow', action='store_true')
    args = parser.parse_args()

    # combine argparse and yaml
    opt = yaml.load(open(args.config), Loader=yaml.FullLoader)
    opt.update(vars(args))
    cfg = opt

    # FORCE NO-FLOW
    cfg['no_flow'] = True

    set_seed(20)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    identifier = f'{cfg["model"]}_{cfg["data_name"]}'
    result_path = create_outdir(osp.join(cfg['output_path'], identifier))
    logger = get_logger(result_path)
    logger.info(cfg)

    testloader = build_data_loader(cfg, mode='validation')

    # model dimension

    torch.cuda.reset_peak_memory_stats(device=None)
    memory = torch.cuda.max_memory_allocated()
    model = build_model(cfg, device)
    memory = torch.cuda.max_memory_allocated()
    print('memory_usage: ', memory / 1e6, '(MB)')

    #elapsed, fps = model.benchmark_framerate()
    #print(f'elapsed ms for frame: {elapsed}, fps: {fps}')

    # load classes weights
    with open(cfg['categories_class_weight'], 'rb') as f:
        weights = [torch.tensor(x, dtype=torch.float32).cuda() 
                    for x in pickle.load(f)]

    evaluate = build_eval(cfg)
    if args.eval != None:
        model.load_state_dict(torch.load(args.eval))
        mAP = evaluate(model, testloader, logger) # NOTE: EVALUATION
        logger.info(f'{cfg["task"]} result: {mAP*100:.2f} m{cfg["metric"]}')
        exit()
        
    trainloader = build_data_loader(cfg, mode='train')
    #criterion = build_criterion(cfg, device)

    train_one_epoch = build_trainer(cfg)
    optim = torch.optim.AdamW if cfg['optimizer'] == 'AdamW' else torch.optim.Adam
    optimizer = optim([{'params': model.parameters(), 'initial_lr': cfg['lr']}],
                        lr=cfg['lr'], weight_decay=cfg["weight_decay"])

    scheduler = build_lr_scheduler(cfg, optimizer, len(trainloader)) if args.lr_scheduler else None
    #writer = SummaryWriter(osp.join(result_path, 'runs')) if args.tensorboard else None
    scaler = torch.cuda.amp.GradScaler() if args.amp else None
    total_params = sum(p.numel() for p in model.parameters())

    logger.info(f'Dataset: {cfg["data_name"]},  Model: {cfg["model"]}')    
    logger.info(f'lr:{cfg["lr"]} | Weight Decay:{cfg["weight_decay"]} | Window Size:{cfg["window_size"]} | Batch Size:{cfg["batch_size"]}') 
    logger.info(f'Total epoch:{cfg["num_epoch"]} | Total Params:{total_params/1e6:.1f} M | Optimizer: {cfg["optimizer"]}')
    logger.info(f'Output Path:{result_path}')

    # TEST PERFORMANCE
    if False:
        blob = torch.zeros((1, 2, 2048)).to('cuda')
        with torch.no_grad():
            t0 = benchmark.Timer(
                stmt='model.forward(blob, None)',
                globals={
                    'model': model,
                    'blob': blob
                }
            )
        print(t0.timeit(100))

    best_quality_score, best_epoch = 1e6, 0
    for epoch in range(1, cfg['num_epoch']+1):

        loss_dict, epoch_loss = train_one_epoch(trainloader, model, None, optimizer, scaler, epoch, 
                                                cfg['alpha'], weights, None, scheduler=scheduler)
        
        logger.info(f'train_loss_total: {epoch_loss}, train_loss: {loss_dict}')
        if epoch_loss < best_quality_score:
            best_quality_score = epoch_loss
            logger.info(f'[LAST BEST!] train_loss_total: {epoch_loss}, train_loss: {loss_dict}')

        epoch_loss_total, epoch_loss, epoch_metrics = evaluate(model, testloader, logger, weights)
        logger.info(f'val_loss_total: {epoch_loss_total}, val_loss: {epoch_loss}, val_metrics: {epoch_metrics}')
