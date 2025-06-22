
#!/usr/bin/env python

import argparse
from torch.autograd import Variable
import tqdm
from dataloaders import fundus_dataloader as DL
from torch.utils.data import DataLoader
from dataloaders import custom_transforms as tr
from torchvision import transforms
from utils.Utils import *
from utils.metrics import *
import networks.deeplabv3 as netd
import networks.deeplabv3_eval as netd_eval
import torch.backends.cudnn as cudnn
import random
import torch.nn.functional as F
import scipy.stats as stats
import torch

# bceloss = torch.nn.BCELoss(reduction='none')
bceloss = torch.nn.BCELoss()
seg_loss = torch.nn.CrossEntropyLoss()
seed = 3377
savefig = False
get_hd = True
model_save = True
if True:
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-file', type=str,default='./source_model/source_model.pth.tar')
    parser.add_argument('--dataset', type=str, default='Domain2')
    parser.add_argument('--source', type=str, default='Domain3')
    parser.add_argument('-g', '--gpu', type=int, default=0)
    parser.add_argument('--data-dir', default='./Data/Fundus')
    parser.add_argument('--out-stride',type=int,default=16)
    parser.add_argument('--sync-bn',type=bool,default=True)
    parser.add_argument('--freeze-bn',type=bool,default=False)
    parser.add_argument('--pseudo-label-threshold', type=float, default=0.75)
    parser.add_argument('--mean-loss-calc-bound-ratio', type=float, default=0.2)
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    model_file = args.model_file

    # 1. dataset
    composed_transforms_train = transforms.Compose([
        tr.Resize(512),
        tr.add_salt_pepper_noise(),
        tr.adjust_light(),
        tr.eraser(),
        tr.Normalize_tf(),
        tr.ToTensor()
    ])
    composed_transforms_test = transforms.Compose([
        tr.Resize(512),
        tr.Normalize_tf(),
        tr.ToTensor()
    ])
    db_train = DL.FundusSegmentation(base_dir=args.data_dir, dataset=args.dataset, split='train/ROIs', transform=composed_transforms_train)
    db_test = DL.FundusSegmentation(base_dir=args.data_dir, dataset=args.dataset, split='test/ROIs', transform=composed_transforms_test)
    db_source = DL.FundusSegmentation(base_dir=args.data_dir, dataset=args.source, split='train/ROIs', transform=composed_transforms_test)

    train_loader = DataLoader(db_train, batch_size=8, shuffle=False, num_workers=1)
    test_loader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)
    source_loader = DataLoader(db_source, batch_size=1, shuffle=False, num_workers=1)

    # 2. model
    model = netd.DeepLab(num_classes=2, backbone='mobilenet', output_stride=args.out_stride, sync_bn=args.sync_bn, freeze_bn=args.freeze_bn) #创建一个model的深度学习实例对象，使用mobilenet作为主干网络
    model_eval = netd_eval.DeepLab(num_classes=2, backbone='mobilenet', output_stride=args.out_stride, sync_bn=args.sync_bn, freeze_bn=args.freeze_bn).cuda()

    if torch.cuda.is_available():
        model = model.cuda()
    print('==> Loading %s model file: %s' %
          (model.__class__.__name__, model_file))
    checkpoint = torch.load(model_file)

    model.load_state_dict(checkpoint['model_state_dict'])

    if args.dataset=="Domain2":
        npfilename = './pseudolabel/ours/pseudolabel_D2.npz'

    elif args.dataset=="Domain1":
        npfilename = './pseudolabel/ours/pseudolabel_D1.npz'

    npdata = np.load(npfilename, allow_pickle=True)

    pseudo_label_dic = npdata['arr_0'].item()
    uncertain_dic = npdata['arr_1'].item()
    importance_map_dic = npdata['arr_2'].item()

    var_list = model.named_parameters()

    optim_gen = torch.optim.Adam(model.parameters(), lr=0.002, betas=(0.9, 0.99))
    best_val_cup_dice = 0.0
    best_val_disc_dice = 0.0
    best_val_cup_dice_std = 0.0
    best_val_disc_dice_std = 0.0
    best_avg = 0.0

    iter_num = 0

    for epoch_num in tqdm.tqdm(range(20), ncols=70):
        not_cup_loss_sum = torch.FloatTensor([0]).cuda()
        cup_loss_sum = torch.FloatTensor([0]).cuda()
        not_cup_loss_num = 0
        cup_loss_num = 0
        not_disc_loss_sum = torch.FloatTensor([0]).cuda()
        disc_loss_sum = torch.FloatTensor([0]).cuda()
        not_disc_loss_num = 0
        disc_loss_num = 0


        lower_bound = args.pseudo_label_threshold * args.mean_loss_calc_bound_ratio
        upper_bound = 1 - ((1 - args.pseudo_label_threshold) * args.mean_loss_calc_bound_ratio)


        for batch_idx, (sample) in enumerate(train_loader):
            data, target, img_name = sample['image'], sample['map'], sample['img_name']
            if torch.cuda.is_available():
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)

            prediction, _, feature = model(data)
            prediction = torch.sigmoid(prediction)

            pseudo_label = [pseudo_label_dic.get(key) for key in img_name]
            pseudo_label = torch.from_numpy(np.asarray(pseudo_label).astype(float)).float().cuda()

            pseudo_label[pseudo_label > 0.75] = 1
            pseudo_label[pseudo_label <= 0.75] = 0

            importance_map = [importance_map_dic.get(key) for key in img_name]
            importance_map = torch.from_numpy(np.asarray(importance_map).astype(float)).float().cuda()

            std_map = [uncertain_dic.get(key) for key in img_name]
            std_map = torch.from_numpy(np.asarray(std_map).astype(float)).float().cuda()

            # distance map
            target_0_obj = F.interpolate(pseudo_label[:, 0:1, ...], size=feature.size()[2:], mode='nearest')
            target_1_obj = F.interpolate(pseudo_label[:, 1:, ...], size=feature.size()[2:], mode='nearest')
            prediction_small = F.interpolate(prediction, size=feature.size()[2:], mode='bilinear', align_corners=True)
            std_map_small = F.interpolate(std_map, size=feature.size()[2:], mode='bilinear', align_corners=True)
            target_0_bck = 1.0 - target_0_obj
            target_1_bck = 1.0 - target_1_obj

            mask_0_obj = torch.zeros([std_map_small.shape[0], 1, std_map_small.shape[2], std_map_small.shape[3]]).cuda()
            mask_0_bck = torch.zeros([std_map_small.shape[0], 1, std_map_small.shape[2], std_map_small.shape[3]]).cuda()
            mask_1_obj = torch.zeros([std_map_small.shape[0], 1, std_map_small.shape[2], std_map_small.shape[3]]).cuda()
            mask_1_bck = torch.zeros([std_map_small.shape[0], 1, std_map_small.shape[2], std_map_small.shape[3]]).cuda()
            mask_0_obj[std_map_small[:, 0:1, ...] < 0.05] = 1.0
            mask_0_bck[std_map_small[:, 0:1, ...] < 0.05] = 1.0
            mask_1_obj[std_map_small[:, 1:, ...] < 0.05] = 1.0
            mask_1_bck[std_map_small[:, 1:, ...] < 0.05] = 1.0
            mask_0 = mask_0_obj + mask_0_bck
            mask_1 = mask_1_obj + mask_1_bck
            mask = torch.cat((mask_0, mask_1), dim=1)

            feature_0_obj = feature * target_0_obj * mask_0_obj
            feature_1_obj = feature * target_1_obj * mask_1_obj
            feature_0_bck = feature * target_0_bck * mask_0_bck
            feature_1_bck = feature * target_1_bck * mask_1_bck

            centroid_0_obj = torch.sum(feature_0_obj * prediction_small[:, 0:1, ...], dim=[0, 2, 3], keepdim=True)
            centroid_1_obj = torch.sum(feature_1_obj * prediction_small[:, 1:, ...], dim=[0, 2, 3], keepdim=True)
            centroid_0_bck = torch.sum(feature_0_bck * (1.0 - prediction_small[:, 0:1, ...]), dim=[0, 2, 3],keepdim=True)
            centroid_1_bck = torch.sum(feature_1_bck * (1.0 - prediction_small[:, 1:, ...]), dim=[0, 2, 3],keepdim=True)

            target_0_obj_cnt = torch.sum(mask_0_obj * target_0_obj * prediction_small[:, 0:1, ...], dim=[0, 2, 3], keepdim=True)
            target_1_obj_cnt = torch.sum(mask_1_obj * target_1_obj * prediction_small[:, 1:, ...], dim=[0, 2, 3],keepdim=True)
            target_0_bck_cnt = torch.sum(mask_0_bck * target_0_bck * (1.0 - prediction_small[:, 0:1, ...]),dim=[0, 2, 3], keepdim=True)
            target_1_bck_cnt = torch.sum(mask_1_bck * target_1_bck * (1.0 - prediction_small[:, 1:, ...]),dim=[0, 2, 3], keepdim=True)

            epsilon = 1e-8
            centroid_0_obj /= (target_0_obj_cnt + epsilon)
            centroid_1_obj /= (target_1_obj_cnt + epsilon)
            centroid_0_bck /= (target_0_bck_cnt + epsilon)
            centroid_1_bck /= (target_1_bck_cnt + epsilon)

            distance_0_obj = torch.sum(torch.pow(feature - centroid_0_obj, 2), dim=1, keepdim=True)
            distance_0_bck = torch.sum(torch.pow(feature - centroid_0_bck, 2), dim=1, keepdim=True)
            distance_1_obj = torch.sum(torch.pow(feature - centroid_1_obj, 2), dim=1, keepdim=True)
            distance_1_bck = torch.sum(torch.pow(feature - centroid_1_bck, 2), dim=1, keepdim=True)


            distance_0_obj = F.interpolate(distance_0_obj, size=data.size()[2:], mode='nearest')
            distance_0_bck = F.interpolate(distance_0_bck, size=data.size()[2:], mode='nearest')
            distance_1_obj = F.interpolate(distance_1_obj, size=data.size()[2:], mode='nearest')
            distance_1_bck = F.interpolate(distance_1_bck, size=data.size()[2:], mode='nearest')

            distance_obj = torch.cat((distance_0_obj, distance_1_obj), dim=1)
            distance_bck = torch.cat((distance_0_bck, distance_1_bck), dim=1)

            threshold_low = 0.0001 #0.02
            low_uncertainty = std_map <= threshold_low
            low_uncertainty_float = low_uncertainty.float()
            low_uncertainty_front_map = low_uncertainty_float * pseudo_label
            low_uncertainty_back_map = low_uncertainty_float * (1 - pseudo_label)

            low_uncertainty_front_distances = distance_obj * low_uncertainty_front_map
            filtered_distances_obj = low_uncertainty_front_distances[low_uncertainty_front_distances != 0].detach().cpu().numpy()
            low_uncertainty_back_distances = distance_bck * low_uncertainty_back_map
            filtered_distances_bck = low_uncertainty_back_distances[low_uncertainty_back_distances != 0].detach().cpu().numpy()

            low_uncertainty_front = std_map * low_uncertainty_front_map
            filtered_low_uncertainty_obj = low_uncertainty_front[low_uncertainty_front != 0].detach().cpu().numpy()
            low_uncertainty_back = std_map * low_uncertainty_back_map
            filtered_low_uncertainty_bck = low_uncertainty_back[low_uncertainty_back != 0].detach().cpu().numpy()

            # KL
            epsilon = 1e-6
            distance_density_obj = stats.gaussian_kde(filtered_distances_obj)
            confidence_density_obj = stats.gaussian_kde(filtered_low_uncertainty_obj)
            support_obj = np.linspace(min(filtered_distances_obj.min(), filtered_low_uncertainty_obj.min()),
                                  max(filtered_distances_obj.max(), filtered_low_uncertainty_obj.max()),
                                  num=1000)
            distance_pdf_obj = torch.tensor(distance_density_obj(support_obj), dtype=torch.float)
            confidence_pdf_obj = torch.tensor(confidence_density_obj(support_obj), dtype=torch.float)
            distance_pdf_obj = torch.clamp(distance_pdf_obj, min=epsilon)
            confidence_pdf_obj = torch.clamp(confidence_pdf_obj, min=epsilon)
            kl_div_obj = torch.nn.functional.kl_div(confidence_pdf_obj.log(), distance_pdf_obj, reduction='sum')

            distance_density_bck = stats.gaussian_kde(filtered_distances_bck)
            confidence_density_bck = stats.gaussian_kde(filtered_low_uncertainty_bck)
            support_bck = np.linspace(min(filtered_distances_bck.min(), filtered_low_uncertainty_bck.min()),
                                      max(filtered_distances_bck.max(), filtered_low_uncertainty_bck.max()),
                                      num=1000)
            distance_pdf_bck = torch.tensor(distance_density_bck(support_bck), dtype=torch.float)
            confidence_pdf_bck = torch.tensor(confidence_density_bck(support_bck), dtype=torch.float)
            distance_pdf_bck = torch.clamp(distance_pdf_bck, min=epsilon)
            confidence_pdf_bck = torch.clamp(confidence_pdf_bck, min=epsilon)
            kl_div_bck = torch.nn.functional.kl_div(confidence_pdf_bck.log(), distance_pdf_bck, reduction='sum')

            for param in model.parameters():
                param.requires_grad = True

            not_cup_loss_sum_batch = torch.sum(-torch.log(1 - prediction[:, 0, ...][(prediction[:, 0, ...] < args.pseudo_label_threshold) * (prediction[:, 0, ...] > lower_bound)]))
            not_cup_loss_num_batch = torch.sum((prediction[:, 0, ...] < args.pseudo_label_threshold) * (prediction[:, 0, ...] > lower_bound))
            cup_loss_sum_batch = torch.sum(-torch.log(prediction[:, 0, ...][(prediction[:, 0, ...] > args.pseudo_label_threshold) * (prediction[:, 0, ...] < upper_bound)]))
            cup_loss_num_batch = torch.sum((prediction[:, 0, ...] > args.pseudo_label_threshold) * (prediction[:, 0, ...] < upper_bound))

            not_disc_loss_sum_batch = torch.sum(-torch.log(1 - prediction[:, 1, ...][(prediction[:, 1, ...] < args.pseudo_label_threshold) * (prediction[:, 1, ...] > lower_bound)]))
            not_disc_loss_num_batch = torch.sum((prediction[:, 1, ...] < args.pseudo_label_threshold) * (prediction[:, 1, ...] > lower_bound))
            disc_loss_sum_batch = torch.sum(-torch.log(prediction[:, 1, ...][(prediction[:, 1, ...] > args.pseudo_label_threshold) * (prediction[:, 1, ...] < upper_bound)]))
            disc_loss_num_batch = torch.sum((prediction[:, 1, ...] > args.pseudo_label_threshold) * (prediction[:, 1, ...] < upper_bound))

            loss_cup_weight_batch = (cup_loss_sum_batch.item() / (cup_loss_num_batch + 1e-6)) / (not_cup_loss_sum_batch.item() / (not_cup_loss_num_batch + 1e-6))
            loss_disc_weight_batch = (disc_loss_sum_batch.item() / (disc_loss_num_batch + 1e-6)) / (not_disc_loss_sum_batch.item() / (not_disc_loss_num_batch + 1e-6))

            max_loss_weight = max(loss_cup_weight_batch, loss_disc_weight_batch)
            min_loss_weight = min(loss_cup_weight_batch, loss_disc_weight_batch)

            if max_loss_weight > 5:
                loss_cup_weight_batch *= 0.9
                loss_disc_weight_batch *= 1.1
            elif min_loss_weight < 0.2:
                loss_cup_weight_batch *= 1.1
                loss_disc_weight_batch *= 0.9

            mean_loss_weight_mask = torch.ones(pseudo_label.size()).cuda()
            mean_loss_weight_mask[:, 0, ...][pseudo_label[:, 0, ...] == 0] = loss_cup_weight_batch
            mean_loss_weight_mask[:, 1, ...][pseudo_label[:, 1, ...] == 0] = loss_disc_weight_batch
            loss_mask = mean_loss_weight_mask

            mean_weight_map = torch.ones(pseudo_label.size()).cuda()
            mean_weight_map[:, 0, ...] = loss_cup_weight_batch
            mean_weight_map[:, 1, ...] = loss_disc_weight_batch
            weight_map = mean_weight_map

            batchsize, channel, H, W = data.shape
            total_pixels = batchsize * H * W
            loss_cb = torch.mean(bceloss(prediction, pseudo_label) * loss_mask )
            loss = torch.mean(loss_cb * importance_map + (kl_div_obj + kl_div_bck))

            loss.backward()
            optim_gen.step()
            optim_gen.zero_grad()
            iter_num += 1

        #test
        model_eval.eval()
        pretrained_dict = model.state_dict()
        model_dict = model_eval.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_eval.load_state_dict(pretrained_dict)

        val_cup_dice = 0.0; val_disc_dice = 0.0; datanum_cnt = 0.0
        cup_hd = 0.0; disc_hd = 0.0; datanum_cnt_cup = 0.0; datanum_cnt_disc = 0.0

        all_cup_dices = []
        all_disc_dices = []
        all_cup_hds = []
        all_disc_hds = []

        with torch.no_grad():
            for batch_idx, (sample) in enumerate(test_loader):
                data, target, img_name = sample['image'], sample['map'], sample['img_name']
                if torch.cuda.is_available():
                    data, target = data.cuda(), target.cuda()
                data, target = Variable(data), Variable(target)
                prediction, boundary, _ = model_eval(data)
                prediction = torch.sigmoid(prediction)

                target_numpy = target.data.cpu()
                prediction = prediction.data.cpu()
                prediction[prediction > 0.75] = 1; prediction[ prediction <= 0.75] = 0

                cup_dice = dice_coefficient_numpy(prediction[:, 0, ...], target_numpy[:, 0, ...])
                disc_dice = dice_coefficient_numpy(prediction[:, 1, ...], target_numpy[:, 1, ...])
                all_cup_dices.append(cup_dice)
                all_disc_dices.append(disc_dice)

                for i in range(prediction.shape[0]):
                    hd_tmp = hd_numpy(prediction[i, 0, ...], target_numpy[i, 0, ...], get_hd)
                    if np.isnan(hd_tmp):
                        datanum_cnt_cup -= 1.0
                    else:
                        cup_hd += hd_tmp
                        all_cup_hds.append(hd_tmp)

                    hd_tmp = hd_numpy(prediction[i, 1, ...], target_numpy[i, 1, ...], get_hd)
                    if np.isnan(hd_tmp):
                        datanum_cnt_disc -= 1.0
                    else:
                        disc_hd += hd_tmp
                        all_disc_hds.append(hd_tmp)

                val_cup_dice += np.sum(cup_dice)
                val_disc_dice += np.sum(disc_dice)

                datanum_cnt += float(prediction.shape[0])
                datanum_cnt_cup += float(prediction.shape[0])
                datanum_cnt_disc += float(prediction.shape[0])

        val_cup_dice /= datanum_cnt
        val_disc_dice /= datanum_cnt
        cup_hd /= datanum_cnt_cup
        disc_hd /= datanum_cnt_disc
        cup_dice_std = np.std(all_cup_dices)
        disc_dice_std = np.std(all_disc_dices)
        cup_hd_std = np.std(all_cup_hds)
        disc_hd_std = np.std(all_disc_hds)

        if (val_cup_dice+val_disc_dice)/2.0>best_avg:
            best_val_cup_dice = val_cup_dice; best_val_disc_dice = val_disc_dice; best_avg = (val_cup_dice+val_disc_dice)/2.0
            best_val_cup_dice_std = cup_dice_std; best_val_disc_dice_std = disc_dice_std
            best_cup_hd = cup_hd; best_disc_hd = disc_hd; best_avg_hd = (best_cup_hd+best_disc_hd)/2.0
            best_cup_hd_std = cup_hd_std; best_disc_hd_std = disc_hd_std

        if not os.path.exists('./logs/train_target_PlainPL'):
            os.mkdir('./logs/train_target_PlainPL')
        if args.dataset == 'Domain1':
            savefile = './logs/train_target_masked_pooling/' + 'D1_' + 'checkpoint_%d.pth.tar' % epoch_num
        elif args.dataset == 'Domain2':
            savefile = './logs/train_target_masked_pooling/' + 'D2_' + 'checkpoint_%d.pth.tar' % epoch_num
        if model_save:
            torch.save({
                'model_state_dict': model.state_dict(),
                'best_mean_dice': best_avg,
                'best_cup_dice': best_val_cup_dice,
                'best_disc_dice': best_val_disc_dice,
                'best_cup_dice_std': best_val_cup_dice_std,
                'best_disc_dice_std': best_val_disc_dice_std
            }, savefile)

        print("cup: %.4f disc: %.4f avg: %.4f cup: %.4f disc: %.4f avg: %.4f" %
              (val_cup_dice, val_disc_dice, (val_cup_dice + val_disc_dice) / 2.0, cup_hd, disc_hd, (cup_hd + disc_hd) / 2.0))
        print("best cup: %.4f cup std: %.4f best disc: %.4f disc std: %.4f best avg: %.4f best cup: %.4f cup std: %.4f best disc: %.4f disc std: %.4f best avg: %.4f" %
              (best_val_cup_dice, best_val_cup_dice_std, best_val_disc_dice, best_val_disc_dice_std, best_avg, best_cup_hd, best_cup_hd_std, best_disc_hd, best_disc_hd_std, best_avg_hd))
        model.train()


