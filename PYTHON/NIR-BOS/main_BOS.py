import argparse
import os

from nerf.provider import NeRFDataset
from nerf.utils import *
import sys
from pathlib import Path
from nerf.mask3D_calculate import Mask3DDataset
from adabelief_pytorch import AdaBelief
import time
from experiment_logging import ExperimentRecorder

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))
from experiment_control.batch_paths import (
    CHECKPOINT_ENV,
    EXTENSION_ENV,
    require_batch_workspace,
)
from experiment_control.training_extension import (
    restart_cosine_scheduler_for_extension,
)

if __name__ == '__main__':
    # Manually assign sys.argv to emulate command-line input
    # 严格控制组 v1：与梯度版人工保持相同的网络主干、训练预算和采样参数。
    sys.argv = ['main_nerf.py', 'data/Phantom 1/140x294x140', '--workspace',
                '__FORMAL_BATCH_WORKSPACE__', '--seed', '0', '--bound', '2', '--fp16',
                '--cuda_ray', '--scale', '0.00054421', '--iters', '30000', '--lr', '5e-3', '--dt_gamma', '0',
                '--num_rays', '256', '--max_steps', '256',
                # 公共评价网格半尺寸：ROInum * ROIvoxelsize / 2。
                '--maskflag','--ROIsize', '0.9523675', '1.99997175', '0.9523675', '--ROInum', '140', '294', '140', '--ROIvoxelsize',
                '0.01360525', '--valbound', '-1.0', '3',
                '--ckpt', 'scratch']#



    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)
    parser.add_argument('-O', action='store_true', help="equals --fp16 --cuda_ray --preload")
    parser.add_argument('--test', action='store_true', help= "test mode")
    parser.add_argument('--workspace', type=str, default='workspace')
    parser.add_argument('--seed', type=int, default=0)

    ### training options
    parser.add_argument('--iters', type=int, default=30000, help="training iters")
    parser.add_argument('--lr', type=float, default=1e-2, help="initial learning rate")
    parser.add_argument('--ckpt', type=str, default='latest')
    parser.add_argument('--num_rays', type=int, default=4096, help="num rays sampled per image for each training step")
    parser.add_argument('--cuda_ray', action='store_true', help="use CUDA raymarching instead of pytorch")
    parser.add_argument('--max_steps', type=int, default=1024,
                        help="max num steps sampled per ray (only valid when using --cuda_ray)")
    parser.add_argument('--num_steps', type=int, default=512,
                        help="num steps sampled per ray (only valid when NOT using --cuda_ray)")
    parser.add_argument('--upsample_steps', type=int, default=0,
                        help="num steps up-sampled per ray (only valid when NOT using --cuda_ray)")
    parser.add_argument('--update_extra_interval', type=int, default=16,
                        help="iter interval to update extra status (only valid when using --cuda_ray)")
    parser.add_argument('--max_ray_batch', type=int, default=4096,
                        help="batch size of rays at inference to avoid OOM (only valid when NOT using --cuda_ray)")
    parser.add_argument('--patch_size', type=int, default=1,
                        help="[experimental] render patches in training, so as to apply LPIPS loss. 1 means disabled, use [64, 32, 16] to enable")

    ### network backbone options
    parser.add_argument('--fp16', action='store_true', help="use amp mixed precision training")
    parser.add_argument('--ff', action='store_true', help="use fully-fused MLP")
    parser.add_argument('--tcnn', action='store_true', help="use TCNN backend")

    ### dataset options
    parser.add_argument('--color_space', type=str, default='srgb', help="Color space, supports (linear, srgb)")
    parser.add_argument('--preload', action='store_true',
                        help="preload all data into GPU, accelerate training but use more GPU memory")
    # (the default value is for the fox dataset)
    parser.add_argument('--bound', type=float, default=2,
                        help="assume the scene is bounded in box[-bound, bound]^3, if > 1, will invoke adaptive ray marching.")
    parser.add_argument('--scale', type=float, default=0.33, help="scale camera location into box[-bound, bound]^3")
    parser.add_argument('--offset', type=float, nargs='*', default=[0, 0, 0], help="offset of camera location")
    parser.add_argument('--dt_gamma', type=float, default=1 / 128,
                        help="dt_gamma (>=0) for adaptive ray marching. set to 0 to disable, >0 to accelerate rendering (but usually with worse quality)")
    parser.add_argument('--min_near', type=float, default=0.2, help="minimum near distance for camera")
    parser.add_argument('--density_thresh', type=float, default=1e-5, help="threshold for density grid to be occupied")


    ### experimental
    parser.add_argument('--error_map', action='store_true', help="use error map to sample rays")
    parser.add_argument('--clip_text', type=str, default='', help="text input for CLIP guidance")
    parser.add_argument('--rand_pose', type=int, default=-1,
                        help="<0 uses no rand pose, =0 only uses rand pose, >0 sample one rand pose every $ known poses")

    ### mask
    parser.add_argument('--maskflag', action='store_true', help="use 3D mask for training")

    ### interest ROI
    parser.add_argument('--ROIsize', type=float, nargs='+', default=[2, 2, 2], help="testing ROI size")
    parser.add_argument('--ROInum', type=int, nargs='+', default=[100, 100, 100], help="testing ROI voxel number")
    parser.add_argument('--ROIvoxelsize', type=float, default=0.0008, help="testing ROI voxel size")

    ### val bound
    parser.add_argument('--valbound', type=float, nargs='+', default=[-1, 0], help="set the val bound")

    opt = parser.parse_args()

    # 正式实验包只动态选择批次保存目录；所有训练参数仍保持上述人工硬编码。
    opt.workspace = str(require_batch_workspace('baseline'))
    sys.argv[sys.argv.index('--workspace') + 1] = opt.workspace
    checkpoint_override = os.environ.get(CHECKPOINT_ENV, '').strip()
    if checkpoint_override:
        # 只供正式控制器恢复同批次 checkpoint；网络与训练参数仍保持硬编码。
        opt.ckpt = checkpoint_override
        sys.argv[sys.argv.index('--ckpt') + 1] = opt.ckpt
    opt.extension_mode = os.environ.get(EXTENSION_ENV, '') == '1'
    if opt.extension_mode and not checkpoint_override:
        raise RuntimeError('延长训练必须由控制器注入父批次 checkpoint')

    if opt.O:
        opt.fp16 = True
        opt.cuda_ray = True
        opt.preload = True

    if opt.patch_size > 1:
        opt.error_map = False  # do not use error_map if use patch-based training
        # assert opt.patch_size > 16, "patch_size should > 16 to run LPIPS loss."
        assert opt.num_rays % (opt.patch_size ** 2) == 0, "patch_size ** 2 should be dividable by num_rays."


    from nerf.network import NeRFNetwork

    print(opt)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mask3Ddata = Mask3DDataset(opt, device=device)


    seed_everything(opt.seed)

    model = NeRFNetwork(
        encoding="Fourier", # or Fourier
        num_layers=3,
        hidden_dim=128,
        bound=opt.bound,
        cuda_ray=opt.cuda_ray,
        density_scale=1,
        min_near=opt.min_near,
        density_thresh=opt.density_thresh,
        mask3Ddata = mask3Ddata,
        ROIsize=opt.ROIsize,
        ROInum=opt.ROInum,
        ROIvoxelsize = opt.ROIvoxelsize,
        valbound = opt.valbound
    )

    print(model)

    experiment_recorder = ExperimentRecorder(
        route='baseline',
        mode='test' if opt.test else 'train',
        opt=opt,
        model=model,
        device=device,
        entrypoint=__file__,
    )

    criterion = torch.nn.MSELoss(reduction='none')




    if opt.test:

        metrics = [PSNRMeter(), LPIPSMeter(device=device)]
        trainer = Trainer('ngp', opt, model, device=device, workspace=opt.workspace, criterion=criterion, fp16=opt.fp16,
                          metrics=metrics, use_checkpoint=opt.ckpt)
        experiment_recorder.mark_trainer_initialized(trainer)


        test_loader = NeRFDataset(opt, device=device, type='test').dataloader()

        if test_loader.has_gt:
            trainer.evaluate(test_loader)  # blender has gt, so evaluate it.

        trainer.test(test_loader, write_video=True)  # test and save video

        trainer.save_mesh(resolution=256, threshold=10)
        experiment_recorder.complete(trainer)

    else:

        optimizer = lambda model: torch.optim.Adam(model.get_params(opt.lr), betas=(0.9, 0.99), eps=1e-8)
        # optimizer = torch.optim.AdamW(model.parameters(), lr=opt.lr, betas=(0.9, 0.999), weight_decay=1e-4)
        # optimizer = lambda model: torch.optim.AdamW(model.get_params(opt.lr), betas=(0.9, 0.999), weight_decay=1e-4, eps=1e-8)
        # optimizer = lambda model: AdaBelief(model.get_params(opt.lr), lr=opt.lr, betas=(0.9, 0.999),eps=1e-8, weight_decay=1e-4,rectify=True)
        # optimizer = lambda model: torch.optim.SGD(model.get_params(opt.lr), lr=opt.lr, momentum=0.9, weight_decay=1e-4, nesterov=True)

        train_loader = NeRFDataset(opt, device=device, type='train').dataloader()

        # decay to 0.1 * init_lr at last iter step
        # scheduler = lambda optimizer: optim.lr_scheduler.LambdaLR(optimizer,
        #                                                           lambda iter: 0.1 ** min(iter / opt.iters, 1)) #初始训练用较大学习率快速收敛,后期用小学习率微调模型
        scheduler = lambda optimizer: optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.iters, eta_min=1e-6)
        # scheduler = lambda optimizer: optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=opt.iters // 10,  T_mult=2, eta_min=1e-6)

        metrics = [PSNRMeter(), LPIPSMeter(device=device)]
        trainer = Trainer('ngp', opt, model, device=device, workspace=opt.workspace, optimizer=optimizer,
                          criterion=criterion, ema_decay=0.95, fp16=opt.fp16, lr_scheduler=scheduler,
                          scheduler_update_every_step=True, metrics=metrics, use_checkpoint=opt.ckpt, eval_interval=50)
        if opt.extension_mode:
            extension_details = restart_cosine_scheduler_for_extension(
                trainer,
                target_iterations=opt.iters,
                restart_learning_rate=opt.lr,
                eta_min=1e-6,
            )
            experiment_recorder.record_extension(extension_details)
        experiment_recorder.mark_trainer_initialized(trainer)


        valid_loader = NeRFDataset(opt, device=device, type='val', downscale=1).dataloader()

        max_epoch = np.ceil(opt.iters / len(train_loader)).astype(np.int32)

        start_time = time.time()
        trainer.train(train_loader, valid_loader, max_epoch)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"total training time: {elapsed_time:.2f} second ({elapsed_time / 60:.2f} minute)")
        save_path = os.path.join(opt.workspace, "train_time.txt")
        with open(save_path, "a") as f:  # 'a' 表示追加模式，不会覆盖旧内容
            f.write(f"total training time: {elapsed_time:.2f} second ({elapsed_time / 60:.2f} minute)\n")
        experiment_recorder.record_training_time(elapsed_time)

        # also test
        test_loader = NeRFDataset(opt, device=device, type='test').dataloader()

        if test_loader.has_gt:
            trainer.evaluate(test_loader)  # blender has gt, so evaluate it.

        trainer.test(test_loader, write_video=True)  # test and save video

        trainer.save_mesh(resolution=256, threshold=0.2)
        experiment_recorder.complete(trainer)
