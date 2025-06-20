import argparse
import matplotlib
matplotlib.use('TkAgg')

from torch.autograd import Variable
import tqdm
from dataloaders import fundus_dataloader as DL
from torch.utils.data import DataLoader
from dataloaders import custom_transforms as tr
from torchvision import transforms
from networks.deeplabv3 import *
import torch.backends.cudnn as cudnn
import random
import os
import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
from torch.nn.functional import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import pandas as pd
import seaborn as sns
matplotlib.use('Agg')  # 设置为Agg后端

bceloss = torch.nn.BCELoss()
seed = 3377
savefig = False
get_hd = False
if True:
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


# 余弦距离
def cosine_distance(a, b):
    """
    计算余弦距离
    :param a: 形状为 (N, D) 的张量，N 是样本数，D 是特征维度
    :param b: 形状为 (1, D) 的张量
    :return: 形状为 (N,) 的张量，表示每个样本到 b 的余弦距离
    """
    # 确保 a 和 b 都是二维张量
    if a.dim() == 1:
        a = a.unsqueeze(0)  # 将一维张量 a 扩展为 (1, D)
    if b.dim() == 1:
        b = b.unsqueeze(0)  # 将一维张量 b 扩展为 (1, D)

    # 计算余弦相似度
    similarity = F.cosine_similarity(a, b, dim=1)  # 在特征维度上计算相似度
    # 计算余弦距离
    distance = 1 - similarity
    return distance
'''

def update_labels_with_features_pre(prediction, current_lowuncer_map, feature_feat, std_map_pre, threshold_high=0.04, radius=5):
    std_map_pre = std_map_pre.cuda()
    current_lowuncer_map = current_lowuncer_map.cuda()

    # Masks for each uncertainty level per class
    std_small_mask = (current_lowuncer_map != 0).cuda()  # [8, 2, 512, 512]
    std_huge_mask = (std_map_pre > threshold_high).cuda()  # [8, 2, 512, 512]
    num_high_uncertainty_pixels = torch.sum(std_huge_mask).item()
    print("Number of high uncertainty pixels:", num_high_uncertainty_pixels)

    #    std_mid_mask = (std_map_pre <= threshold_high) & (current_lowuncer_map == 0).cuda()  # [8, 2, 512, 512]

    feature_feat = feature_feat.cuda()
    feature_feat = F.interpolate(feature_feat, size=(512, 512), mode='bilinear', align_corners=False)
    conv = nn.Conv2d(305, 2, kernel_size=1)
    if torch.cuda.is_available():
        conv = conv.cuda()
    feature_feat = conv(feature_feat)

    updated_prediction = prediction.clone() # [8, 2, 512, 512]

    for c in range(prediction.shape[1]):
        std_huge_indices = torch.nonzero(std_huge_mask[:, c, ...], as_tuple=True)
    #   std_mid_indices = torch.nonzero(std_mid_mask[:, c, ...], as_tuple=True)
        std_small_indices = torch.nonzero(std_small_mask[:, c, ...], as_tuple=True)

        # Validate indices and extract features for low uncertainty pixels
        if std_small_indices[0].numel() > 0:
            valid_small_indices = (std_small_indices[1] < feature_feat.size(2)) & (std_small_indices[2] < feature_feat.size(3))
            low_uncertainty_features = feature_feat[std_small_indices[0][valid_small_indices], c, std_small_indices[1][valid_small_indices],std_small_indices[2][valid_small_indices]]
            # print("low_uncertainty_features:",low_uncertainty_features.shape)
            low_uncertainty_labels = prediction[std_small_indices[0][valid_small_indices], c, std_small_indices[1][valid_small_indices],std_small_indices[2][valid_small_indices]]

        # Process high uncertainty pixels
        for i in range(len(std_huge_indices[0])):
            if std_huge_indices[1][i] < feature_feat.size(2) and std_huge_indices[2][i] < feature_feat.size(3):
                # y:std_huge_indices[1][i] x:std_huge_indices[2][i]
                idx = (std_huge_indices[0][i], c, std_huge_indices[1][i], std_huge_indices[2][i])
                high_uncertainty_feature = feature_feat[idx].unsqueeze(0)

                # Calculate neighborhood mask within the radius
                neighborhood_mask = ((std_small_indices[1] >= std_huge_indices[1][i] - radius) &
                                     (std_small_indices[1] <= std_huge_indices[1][i] + radius) &
                                     (std_small_indices[2] >= std_huge_indices[2][i] - radius) &
                                     (std_small_indices[2] <= std_huge_indices[2][i] + radius)) & valid_small_indices

                # print("neighborhood_mask",neighborhood_mask.shape)

                if torch.any(neighborhood_mask):
                    local_features = low_uncertainty_features[neighborhood_mask]
                    local_labels = low_uncertainty_labels[neighborhood_mask]
                    # print("Local features shape:", local_features.shape)  # 打印距离的形状
                    # print("High uncertainty feature shape:", high_uncertainty_feature.shape)  # 打印距离的形状
                    distances = cosine_distance(local_features, high_uncertainty_feature)
                    # print("Distances shape:", distances.shape)  # 打印距离的形状
                    # print("Distances values:", distances)  # 打印距离的值
                    nearest_index = torch.argmin(distances)
                    updated_prediction[idx] = local_labels[nearest_index]
                    # mean_local_features = torch.mean(local_features, dim=0, keepdim=True)
                    # similarity = torch.cosine_similarity(mean_local_features, high_uncertainty_feature, dim=-1)
                    # if similarity > 0.9:
                        # Use the label from one of the nearest low uncertainty points
                    #     updated_prediction[idx] = local_labels[0]

    return updated_prediction
'''

def euclidean_distance(a, b):
    if a.dim() == 1:
        a = a.unsqueeze(0)  # 将一维张量 a 扩展为 (1, D)
    if b.dim() == 1:
        b = b.unsqueeze(0)  # 将一维张量 b 扩展为 (1, D)
    # 计算欧几里得距离
    return torch.sqrt(torch.sum((a - b) ** 2, dim=1))

def masked_average_pooling(prediction, features, low_uncertainty_map, target_size):
    # 为每个类别单独进行操作
    num_classes = low_uncertainty_map.size(1)  # 这里假设 low_uncertainty_map 的形状为 [batch_size, num_classes, height, width]
    class_centers = []
    low_uncertainty_map = low_uncertainty_map.float()


    for class_index in range(num_classes):
        # 上采样当前类别的低不确定性地图
        class_map = low_uncertainty_map[:, class_index:class_index+1, :, :]  # torch.Size([8, 1, 512, 512])
        resized_map = F.interpolate(class_map, size=target_size, mode='bilinear', align_corners=False) # torch.Size([8, 1, 128, 128])
        # 扩展到特征图的通道数
        extended_map = resized_map.expand(-1, features.size(1), -1, -1)  # torch.Size([8, 305, 128, 128])

        class_prediction = prediction[:, class_index:class_index+1, :, :]
        prediction_small = F.interpolate(class_prediction, size=feature.size()[2:], mode='bilinear', align_corners=True)
        prediction_small = prediction_small.expand(-1, features.size(1), -1, -1)

        # 应用掩模并计算特征中心
        # print(features.shape)
        # print(extended_map.shape)
        # print(prediction_small.shape)
        masked_features = features * extended_map * prediction_small
        sum_mask = torch.sum(extended_map * prediction_small , dim=[2, 3])  # 求和以避免除以零
        sum_mask = torch.where(sum_mask == 0, torch.ones_like(sum_mask), sum_mask)  # 避免除以零
        class_center = torch.sum(masked_features, dim=[2, 3]) / sum_mask  # 形状 [batch_size, num_features]
        class_centers.append(class_center)

    # 将所有类别的中心叠加到一个张量中
    low_uncertainty_centers = torch.stack(class_centers, dim=1)  # torch.Size([8, 2, 305])

    return low_uncertainty_centers

def update_labels_with_features(prediction, current_lowuncer_map, feature_feat, std_map_pre, threshold_high=0.05, radius=5):
    std_map_pre = std_map_pre.cuda()
    current_lowuncer_map = current_lowuncer_map.cuda()

    std_small_mask = (current_lowuncer_map != 0).cuda()  # 低不确定性掩码
    std_huge_mask = (std_map_pre > threshold_high).cuda()  # 高不确定性掩码

    feature_feat = feature_feat.cuda()
    feature_feat = F.interpolate(feature_feat, size=(512, 512), mode='bilinear', align_corners=False)
    conv = nn.Conv2d(305, 2, kernel_size=1).cuda()
    feature_feat = conv(feature_feat)

    updated_prediction = prediction.clone()  # 复制预测

    for c in range(prediction.shape[1]):
        std_huge_indices = torch.nonzero(std_huge_mask[:, c, ...], as_tuple=True)
        std_small_indices = torch.nonzero(std_small_mask[:, c, ...], as_tuple=True)

        if std_small_indices[0].numel() > 0:
            valid_small_indices = (std_small_indices[1] < feature_feat.size(2)) & (std_small_indices[2] < feature_feat.size(3))
            low_uncertainty_features = feature_feat[std_small_indices[0][valid_small_indices], c, std_small_indices[1][valid_small_indices], std_small_indices[2][valid_small_indices]]
            low_uncertainty_labels = prediction[std_small_indices[0][valid_small_indices], c, std_small_indices[1][valid_small_indices], std_small_indices[2][valid_small_indices]]

        for i in range(len(std_huge_indices[0])):
            idx = (std_huge_indices[0][i], c, std_huge_indices[1][i], std_huge_indices[2][i])
            high_uncertainty_feature = feature_feat[idx].unsqueeze(0)

            # 计算邻域掩码
            neighborhood_mask = ((std_small_indices[1] >= std_huge_indices[1][i] - radius) &
                                  (std_small_indices[1] <= std_huge_indices[1][i] + radius) &
                                  (std_small_indices[2] >= std_huge_indices[2][i] - radius) &
                                  (std_small_indices[2] <= std_huge_indices[2][i] + radius)) & valid_small_indices
            # print("neighborhood_mask",neighborhood_mask.shape)

            if torch.any(neighborhood_mask):
                local_features = low_uncertainty_features[neighborhood_mask]
                local_labels = low_uncertainty_labels[neighborhood_mask]

                # 计算局部距离
                # distances = cosine_distance(local_features, high_uncertainty_feature)
                distances = euclidean_distance(local_features, high_uncertainty_feature)
                # print("Distances shape:", distances.shape)  # 打印距离的形状
                # print("Distances values:", distances)  # 打印距离的值

                # 动态计算局部平均距离作为阈值
                if distances.numel() > 0:
                    local_mean_distance = distances.mean()
                    threshold_distance = local_mean_distance.item() # 使用局部平均距离的一部分
                    # print("threshold_distance",threshold_distance)

                    # 使用阈值进行标签更新
                    nearest_index = torch.argmin(distances)
                    if distances[nearest_index] < threshold_distance:  # 使用动态计算的阈值
                        updated_prediction[idx] = local_labels[nearest_index]

    return updated_prediction

def plot_maps(map, img_names, save_dir = None):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    updated_img_names = img_names.copy()

    for i in range(map.shape[0]):
        updated_img_names[i] = updated_img_names[i].split('.')[0]
        # print(img_names[i])
        cup_prediction = map[i, 0, ...].cpu().numpy()
        disc_prediction = map[i, 1, ...].cpu().numpy()

        height, width = cup_prediction.shape
        rgb_image_cup = np.zeros((height, width, 3), dtype=np.uint8)
        rgb_image_disc = np.zeros((height, width, 3), dtype=np.uint8)

        rgb_image_cup[:, :] = [0, 0, 0]
        rgb_image_cup[cup_prediction == 1] = [255, 255, 255]

        rgb_image_disc[:, :] = [0, 0, 0]
        rgb_image_disc[disc_prediction == 1] = [255, 255, 255]

        plt.imshow(rgb_image_cup)
        plt.axis('off')
        plt.show()
        plt.savefig(os.path.join(save_dir, f'{updated_img_names[i]}_cup.png'), bbox_inches='tight', pad_inches=0)
        plt.close()

        plt.imshow(rgb_image_disc)
        plt.axis('off')
        plt.savefig(os.path.join(save_dir, f'{updated_img_names[i]}_disc.png'), bbox_inches='tight', pad_inches=0)
        plt.close()

def plot_feature(feature_0, feature_1, img_names, save_dir=None):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    updated_img_names = img_names.copy()

    for i in range(feature_0.shape[0]):
        updated_img_names[i] = updated_img_names[i].split('.')[0]
        feature_0_2d = feature_0[i, ...]
        feature_0_2d = torch.sum(feature_0_2d, dim = 0).cpu().numpy()
        feature_1_2d = feature_1[i, ...]
        feature_1_2d = torch.sum(feature_1_2d, dim=0).cpu().numpy()

        plt.imshow(feature_0_2d, cmap='winter')  # 使用热力图
        plt.axis('off')
        plt.savefig(os.path.join(save_dir, f'{updated_img_names[i]}_cup_feature.png'), bbox_inches='tight', pad_inches=0)
        plt.close()

        plt.imshow(feature_1_2d, cmap='winter')  # 使用热力图
        plt.axis('off')
        plt.savefig(os.path.join(save_dir, f'{updated_img_names[i]}_disc_feature.png'), bbox_inches='tight',pad_inches=0)
        plt.close()

def plot_std_map(std_map, img_names, save_dir=None):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    updated_img_names = img_names.copy()

    for i in range(std_map.shape[0]):
        updated_img_names[i] = updated_img_names[i].split('.')[0]

        cup_std = std_map[i, 0, ...].cpu().numpy()
        disc_std = std_map[i, 1, ...].cpu().numpy()

        plt.imshow(cup_std, cmap='winter')  # 使用热力图
        # plt.colorbar()  # 显示颜色条
        plt.axis('off')
        plt.savefig(os.path.join(save_dir, f'{updated_img_names[i]}_cup_std.png'), bbox_inches='tight', pad_inches=0)
        plt.close()

        plt.imshow(disc_std, cmap='winter')  # 使用热力图
        # plt.colorbar()  # 显示颜色条
        plt.axis('off')
        plt.savefig(os.path.join(save_dir, f'{updated_img_names[i]}_disc_std.png'), bbox_inches='tight', pad_inches=0)
        plt.close()

def plot_entropy_source(entropy_map, img_names, save_dir=None):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    updated_img_names = img_names.copy()

    for i in range(entropy_source.shape[0]):
        updated_img_names[i] = updated_img_names[i].split('.')[0]

        entropy_map_cup = entropy_map[i, 0, ...]
        entropy_map_disc = entropy_map[i, 1, ...]

        entropy_map_normalized_cup = (entropy_map_cup - np.min(entropy_map_cup)) / (np.max(entropy_map_cup) - np.min(entropy_map_cup)) * 255
        entropy_map_normalized_cup = np.uint8(entropy_map_normalized_cup)

        entropy_map_normalized_disc = (entropy_map_disc - np.min(entropy_map_disc)) / (np.max(entropy_map_disc) - np.min(entropy_map_disc)) * 255
        entropy_map_normalized_disc = np.uint8(entropy_map_normalized_disc)

        plt.imshow(entropy_map_normalized_cup, cmap='hot')
        plt.axis('off')  # 不显示坐标轴
        plt.savefig(os.path.join(save_dir, f'{updated_img_names[i]}_entropy_cup.png'), bbox_inches='tight', pad_inches=0)
        plt.close()

        plt.imshow(entropy_map_normalized_disc, cmap='hot')
        plt.axis('off')  # 不显示坐标轴
        plt.savefig(os.path.join(save_dir, f'{updated_img_names[i]}_entropy_disc.png'), bbox_inches='tight', pad_inches=0)
        plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-file', type=str,
                        default='./source_model/source_model.pth.tar')
    parser.add_argument('--dataset', type=str, default='Domain2')
    parser.add_argument('--batchsize', type=int, default=8)
    parser.add_argument('--source', type=str, default='Domain3')
    parser.add_argument('-g', '--gpu', type=int, default=0)
    parser.add_argument('--data-dir', default='./Data/Fundus')
    parser.add_argument('--out-stride', type=int, default=16)
    parser.add_argument('--save-root-ent', type=str, default='./results/ent/')
    parser.add_argument('--save-root-mask', type=str, default='./results/mask/')
    parser.add_argument('--sync-bn', type=bool, default=True)
    parser.add_argument('--freeze-bn', type=bool, default=False)
    parser.add_argument('--test-prediction-save-path', type=str, default='./results/baseline/')

    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    model_file = args.model_file

    # 1. dataset
    composed_transforms_test = transforms.Compose([
        tr.Resize(512),
        tr.Normalize_tf(),
        tr.ToTensor()
    ])
    db_train = DL.FundusSegmentation(base_dir=args.data_dir, dataset=args.dataset, split='train/ROIs', transform=composed_transforms_test)
    db_test = DL.FundusSegmentation(base_dir=args.data_dir, dataset=args.dataset, split='test/ROIs', transform=composed_transforms_test)
    db_source = DL.FundusSegmentation(base_dir=args.data_dir, dataset=args.source, split='train/ROIs', transform=composed_transforms_test)

    train_loader = DataLoader(db_train, batch_size=args.batchsize, shuffle=False, num_workers=1)
    test_loader = DataLoader(db_test, batch_size=args.batchsize, shuffle=False, num_workers=1)
    source_loader = DataLoader(db_source, batch_size=args.batchsize, shuffle=False, num_workers=1)

    total_samples = len(train_loader.dataset)

    # 2. model
    model = DeepLab(num_classes=2, backbone='mobilenet', output_stride=args.out_stride, sync_bn=args.sync_bn, freeze_bn=args.freeze_bn)

    if torch.cuda.is_available():
        model = model.cuda()
    print('==> Loading %s model file: %s' %
          (model.__class__.__name__, model_file))
    checkpoint = torch.load(model_file)

    model.load_state_dict(checkpoint['model_state_dict'])

    model.train()

    pseudo_label_dic = {} # 伪标签字典
    uncertain_dic = {} # 不确定性字典
    importance_map_dic = {} # 重要性图字典

    low_uncertainty_centers_list = []
    processed_samples = 0

    features_list = []
    boundary_list = []
    low_uncertainty_front_map_list = []
    low_uncertainty_back_map_list = []
    all_true_labels = []
    all_img_names = []
    pseudo_label_list = []
    std_map_list = []
    mean_low_uncertainty= torch.zeros(2, 305)

    with torch.no_grad():
        ########### STEP 1 ###########
        for batch_idx, sample in tqdm.tqdm(enumerate(train_loader),
                                           total=len(train_loader),
                                           ncols=80, leave=False):
            data, target, img_name = sample['image'], sample['map'], sample['img_name']

            if torch.cuda.is_available():
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target) # 包装张量torch.Size([8, 3, 512, 512])
            processed_samples += data.size(0)

            all_true_labels.append(target.cpu())  # 将数据移到CPU并存储
            all_img_names.extend(img_name)

            preds = torch.zeros([10, data.shape[0], 2, data.shape[2], data.shape[3]]).cuda() # batchsize, channel, height, weight
            features = torch.zeros([10, data.shape[0], 305, 128, 128]).cuda() # 倒数第二层的channel是305
            boundary = torch.zeros([10, data.shape[0], 2, data.shape[2], data.shape[3]]).cuda()

            for i in range(10):
                with torch.no_grad():
                    preds[i, ...], boundary[i, ...], features[i, ...] = model(data) # feature是倒数第二层输出的结果

            preds1 = torch.sigmoid(preds)
            preds = torch.sigmoid(preds / 2.0)
            std_map = torch.std(preds, dim=0) # torch.Size([8, 2, 512, 512])
            prediction = torch.mean(preds1, dim=0) # torch.Size([8, 2, 512, 512])
            feature = torch.mean(features, dim=0) # torch.Size([8, 305, 128, 128])
            boundaries = torch.mean(boundary, dim=0)

            source_pro_two = torch.zeros(prediction.size()).cuda()
            source_pro_one = torch.zeros(prediction.size()).cuda()
            source_pro_two[prediction < 0.5] = prediction[prediction < 0.5] # 前景小于背景的
            source_pro_one[prediction > 0.5] = prediction[prediction > 0.5] # 前景大于背景的

            prediction_back = 1 - prediction
            source_pro_two[prediction_back < 0.5] = prediction_back[prediction_back < 0.5] # 背景小于前景的
            source_pro_one[prediction_back > 0.5] = prediction_back[prediction_back > 0.5] # 背景大于前景的

            entropy_source = (1. - torch.div(source_pro_two, source_pro_one))
            entropy_source = entropy_source.detach().cpu().numpy()

            for i in range(prediction.shape[0]):
                importance_map_dic[img_name[i]] = entropy_source[i]

            pseudo_label = prediction.clone() # 创建一个张量的副本
            pseudo_label[pseudo_label > 0.75] = 1.0
            pseudo_label[pseudo_label <= 0.75] = 0.0

            PL_0 = F.interpolate(pseudo_label[:, 0:1, ...], size=feature.size()[2:], mode='nearest')
            PL_1 = F.interpolate(pseudo_label[:, 1:, ...], size=feature.size()[2:], mode='nearest')
            feature_0 = feature * PL_0
            feature_1 = feature * PL_1
            # print("feature_0 shape:",feature_0.shape)
            # print("feature_1 shape:", feature_1.shape)

            # plot maps
            '''
            plot_maps(pseudo_label, img_name, save_dir = './pics/initial pseudo label 2')
            plot_feature(feature_0, feature_1, img_name, save_dir='./pics/feature 2')
            plot_std_map(std_map, img_name, save_dir='./pics/std map 2')
            plot_entropy_source(entropy_source, img_name, save_dir='./pics/importance map 2')
            '''


            pseudo_label_list.append(pseudo_label)
            features_list.append(feature)
            std_map_list.append(std_map)
            boundary_list.append(boundaries)

            threshold_low = 0.02
            low_uncertainty = (std_map <= threshold_low)
            low_uncertainty_float = low_uncertainty.float()  # 将布尔型转换为浮点型
            low_uncertainty_front_map = low_uncertainty_float * pseudo_label  # 计算低不确定性前景图
            low_uncertainty_back_map = low_uncertainty_float * (1 - pseudo_label)

            low_uncertainty_front_map_list.append(low_uncertainty_front_map)
            low_uncertainty_back_map_list.append(low_uncertainty_back_map)

            low_uncertainty_centers = masked_average_pooling(prediction, feature, low_uncertainty_front_map, target_size=(128, 128))
            low_uncertainty_centers_list.append(low_uncertainty_centers)

            all_low_uncertainty_centers_tensor = torch.cat(low_uncertainty_centers_list, dim=0)

            if processed_samples == total_samples:
                transposed_tensor = torch.transpose(all_low_uncertainty_centers_tensor, 0,1)  # torch.Size([2, 50, 305])
                # 分别得到了两个类别的中心特征
                mean_low_uncertainty = torch.mean(transposed_tensor, dim=1) # torch.Size([2, 305])

        print("STEP 1 FINISHED")

        ########### STEP 2 ###########
        sample_index = 0
        adjusted_maps = torch.zeros(total_samples, 2, 128, 128)  # 初始化调整地图

        for batch_idx, sample in tqdm.tqdm(enumerate(train_loader), total=len(train_loader), ncols=80, leave=False):
            data, target, img_name = sample['image'], sample['map'], sample['img_name']
            if torch.cuda.is_available():
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)  # torch.Size([8, 3, 512, 512])

            num_classes = low_uncertainty_float.size(1)
            num_samples = data.size(0)

            features_pre = features_list[batch_idx]

            for class_index in range(num_classes):
                # 提取当前类别的低不确定性地图
                current_class_center = mean_low_uncertainty[class_index, :].unsqueeze(1).unsqueeze(2)  # 形状 [305, 1, 1]
                current_class_center = current_class_center.expand(-1, 128, 128).cuda()
                for i in range(num_samples):
                    current_class_feature = features_pre[i, :, :, :].cuda()  # torch.Size([305, 128, 128])
                    similarity = cosine_similarity(current_class_feature, current_class_center,dim=0) # torch.Size([128, 128])

                    adjusted_maps[sample_index + i, class_index, :, :] = similarity
                    adjusted_maps[sample_index + i, 1, :, :] = torch.max(adjusted_maps[sample_index + i, 0, :, :],
                                                                         adjusted_maps[sample_index + i, 1, :, :])

            sample_index += num_samples

        # 每个位置的特征图与相应类别中心特征的余弦相似度
        # 数值大小直接代表了每个位置的特征与类中心特征的相似度。较高的值（接近1）表示较高的相似度
        adjusted_maps = F.interpolate(adjusted_maps, size=(512, 512), mode='bilinear', align_corners=False)  # torch.Size([50, 2, 512, 512])


        ########### STEP 3 ###########
        print("STEP 2 FINISHED")
        sample_index = 0
        new_lowuncer_map = torch.zeros(total_samples, 2, 512, 512)
        confidence_threshold = 0.7
        adjusted_maps[adjusted_maps < confidence_threshold] = 0

        for batch_idx, sample in tqdm.tqdm(enumerate(train_loader), total=len(train_loader), ncols=80, leave=False):
            data, target, img_name = sample['image'], sample['map'], sample['img_name']
            if torch.cuda.is_available():
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)  # torch.Size([8, 3, 512, 512])

            num_classes = low_uncertainty_float.size(1)
            num_samples = data.size(0)

            low_uncertainty_map = low_uncertainty_front_map_list[batch_idx].cuda()  # torch.Size([8, 2, 512, 512])

            for class_index in range(num_classes):
                for i in range(num_samples):
                    new_lowuncer_map[sample_index + i, class_index, :, :] = low_uncertainty_map[i, class_index, :, :] * adjusted_maps[i, class_index, :, :].float().cuda()

            sample_index += num_samples

        print("STEP 3 FINISHED")

        # 此时已经得到了low_uncertainty_map [50,2,512,512]

        ########### STEP 4 ###########
        for batch_idx, sample in tqdm.tqdm(enumerate(train_loader), total=len(train_loader), ncols=80, leave=False):
            data, target, img_name = sample['image'], sample['map'], sample['img_name']
            if torch.cuda.is_available():
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)  # torch.Size([8, 3, 512, 512])

            # print("img_name 2", img_name)

            num_classes = low_uncertainty_float.size(1)
            num_samples = data.size(0)

            features_pre = features_list[batch_idx] #[8, 305, 128, 128]
            pseudo_label_pre = pseudo_label_list[batch_idx] # [8, 2, 512, 512]
            std_map_pre = std_map_list[batch_idx] # [8, 2, 512, 512]
            boundary_pre = boundary_list[batch_idx]
            # 计算std_map_pre的统计数据

            current_lowuncer_back_map = low_uncertainty_back_map_list[batch_idx].cuda() # [8, 2, 512, 512]

            start_index = batch_idx * args.batchsize  # 正确使用每批次的索引计算
            end_index = start_index + num_samples
            current_lowuncer_front_map = new_lowuncer_map[start_index:end_index, :, :, :].cuda()  # [8, 2, 512, 512]

            current_lowuncer_map = current_lowuncer_front_map + current_lowuncer_back_map

            # 伪标签修正
            pseudo_label = update_labels_with_features(pseudo_label_pre, current_lowuncer_map, features_pre, std_map_pre)
            # plot_maps(pseudo_label, img_name, save_dir='./pics/calibrated pseudo label 2')

            pseudo_label = pseudo_label.detach().cpu().numpy()
            std_map_pre = std_map_pre.detach().cpu().numpy()

            for i in range(pseudo_label_pre.shape[0]):
                pseudo_label_dic[img_name[i]] = pseudo_label[i]
                uncertain_dic[img_name[i]] = std_map_pre[i]

            if args.dataset == "Domain1":
                np.savez('./pseudolabel/ours/pseudolabel_D1', pseudo_label_dic, uncertain_dic, importance_map_dic)

            elif args.dataset == "Domain2":
                np.savez('./pseudolabel/ours/pseudolabel_D2_pre', pseudo_label_dic, uncertain_dic, importance_map_dic)
