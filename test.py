import os
import argparse
import numpy as np
import torch
import random
from torch.utils.data import DataLoader
from utils.config import get_config
from utils.evaluation import get_eval
from models.model_dict import get_model
from utils.data_us import JointTransform2D, ImageToImage2D
from utils.loss_functions.sam_loss import get_criterion
from thop import profile


os.environ['CUDA_VISIBLE_DEVICES'] = '7'


def main():

    # ========== parameters setting ==========

    parser = argparse.ArgumentParser(description='Networks')
    parser.add_argument('--modelname', default='BUSSAM', type=str, help='type of model, e.g., SAM, SAMFull, SAMHead, MSA, SAMed, BUSSAM')
    parser.add_argument('--encoder_input_size', type=int, default=256, help='the image size of the encoder input, 1024 in SAM and MSA, 512 in SAMed, 256 in BUSSAM')
    parser.add_argument('--low_image_size', type=int, default=128, help='the image embedding size, 256 in SAM and MSA, 128 in SAMed and BUSSAM')
    parser.add_argument('--task', default='BUSI', help='task or dataset name, e.g., AMUBUS, BUSI')
    parser.add_argument('--vit_name', type=str, default='vit_b', help='select the vit model for the image encoder of sam')
    parser.add_argument('--sam_ckpt', type=str, default='checkpoints/sam_vit_b_01ec64.pth', help='Pretrained checkpoint of SAM')
    parser.add_argument('--batch_size', type=int, default=8, help='batch_size per gpu')
    parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')

    args = parser.parse_args()
    opt = get_config(args.task)
    opt.mode = 'test'
    opt.visual = True
    device = torch.device(opt.device)

    # ========== add the seed to make sure the results are reproducible ==========

    seed_value = 300  # the number of seed
    np.random.seed(seed_value)  # set random seed for numpy
    random.seed(seed_value)  # set random seed for python
    os.environ['PYTHONHASHSEED'] = str(seed_value)  # avoid hash random
    torch.manual_seed(seed_value)  # set random seed for CPU
    torch.cuda.manual_seed(seed_value)  # set random seed for one GPU
    torch.cuda.manual_seed_all(seed_value)  # set random seed for all GPU
    torch.backends.cudnn.deterministic = True  # set random seed for convolution

    # ========== model and data preparation ==========

    # register the sam model
    opt.batch_size = args.batch_size * args.n_gpu

    tf_val = JointTransform2D(img_size=args.encoder_input_size, low_img_size=args.low_image_size, ori_size=opt.img_size, crop=opt.crop, p_flip=0, color_jitter_params=None, long_mask=True)
    test_dataset = ImageToImage2D(opt.data_path, opt.test_split, tf_val, img_size=args.encoder_input_size, class_id=1)
    testloader = DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=8, pin_memory=True)

    if args.modelname == 'SAMed':
        opt.classes = 2
    model = get_model(args.modelname, args=args, opt=opt)
    model.to(device)
    model.train()

    checkpoint = torch.load(opt.load_path)
    # when the load model is saved under multiple GPU
    new_state_dict = {}
    for k, v in checkpoint.items():
        if k[:7] == 'module.':
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    model.load_state_dict(new_state_dict)

    criterion = get_criterion(modelname=args.modelname, opt=opt)

    # ========== begin to test the model ==========

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Total_params:{}'.format(pytorch_total_params))
    input = torch.randn(1, 1, args.encoder_input_size, args.encoder_input_size).cuda()
    points = (torch.tensor([[[1, 2]]]).float().cuda(), torch.tensor([[1]]).float().cuda())
    flops, params = profile(model, inputs=(input, points), )
    print('Gflops:{}, params:{}'.format(flops / 1000000000, params))

    # ---------- testing ----------
    model.eval()
    mean_dice, mean_hdis, mean_iou, mean_acc, mean_se, mean_sp, std_dice, std_hdis, std_iou, std_acc, std_se, std_sp = get_eval(testloader, model, criterion=criterion, opt=opt, args=args)
    print('model_name:{}, task:{}, checkpoint:{}'.format(args.modelname, args.task, opt.load_path))
    print('mean_acc:{:.5f}, mean_se:{:.5f}, mean_dice:{:.5f}, mean_iou:{:.5f}, mean_hdis:{:.5f}'.format(mean_acc[1], mean_se[1], mean_dice[1], mean_iou[1], mean_hdis[1]))

    # write test results to 'test_results.txt'
    with open(opt.output_path + 'test_results.txt', 'a+') as file:
        file.write('model_name:' + args.modelname + ', task:' + args.task + '\n' + 'checkpoint:' + opt.load_path + '\n')
        file.write('Acc:%.5f' % (mean_acc[1]) + ', ')
        file.write('Se:%.5f' % (mean_se[1]) + ', ')
        file.write('Dice:%.5f' % (mean_dice[1]) + ', ')
        file.write('IoU:%.5f' % (mean_iou[1]) + ', ')
        file.write('HD:%.5f' % (mean_hdis[1]) + '\n')
        file.write('-' * 100 + '\n')
        file.close()


if __name__ == '__main__':
    main()
