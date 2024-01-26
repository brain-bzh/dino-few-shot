import torch
from PIL import Image
import torchvision.transforms as transforms
import os
import matplotlib
import matplotlib.pyplot as plt
from utils import visualize_few_shot_space
from utils import hover_few_shot_space

shots = 18
queries = 2

# torch.set_default_device('cuda')

# dinov2_vitl14_reg = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14_reg')

inputs = []

transform = transforms.Compose([
    transforms.Resize(256, antialias=True),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])



i = 0
inputs = []

    # def get_class_images(class_name):
    #     class_images = []
    #     for image_path in sorted(os.listdir('images/' + class_name)):
    #         image = Image.open('images/' + class_name + '/' + image_path)
    #         image = transform(image)
    #         class_images.append(image)
    #     class_images = torch.stack(class_images).to('cuda')
    #     return class_images

    # class_f_images = get_class_images('f')
    # class_k_images = get_class_images('k')
    # class_p_images = get_class_images('p')
    # class_r_images = get_class_images('r')

# class_f_features = dinov2_vitl14_reg(class_f_images)
# class_k_features = dinov2_vitl14_reg(class_k_images)
# class_p_features = dinov2_vitl14_reg(class_p_images)
# class_r_features = dinov2_vitl14_reg(class_r_images)

# # save features
# torch.save(class_f_features, 'class_f_features.pt')
# torch.save(class_k_features, 'class_k_features.pt')
# torch.save(class_p_features, 'class_p_features.pt')
# torch.save(class_r_features, 'class_r_features.pt')

# load features
class_f_features = torch.load('features/class_f_features.pt', map_location=torch.device('cpu'))
class_k_features = torch.load('features/class_k_features.pt', map_location=torch.device('cpu'))
class_p_features = torch.load('features/class_p_features.pt', map_location=torch.device('cpu'))
class_r_features = torch.load('features/class_r_features.pt', map_location=torch.device('cpu'))
print("class_f_features_load: ", class_f_features.shape)

def norm_center_norm(features):
    # L2 normalization
    features = features / features.norm(dim=-1, keepdim=True)
    # centering
    features = features - features.mean(dim=-1, keepdim=True)
    # L2 normalization
    features = features / features.norm(dim=-1, keepdim=True)
    return features

class_f_features = norm_center_norm(class_f_features)
class_k_features = norm_center_norm(class_k_features)
class_p_features = norm_center_norm(class_p_features)
class_r_features = norm_center_norm(class_r_features)

# Nearest Class Mean
class_f_mean = class_f_features[0:shots-1].mean(dim=0)
class_k_mean = class_k_features[0:shots-1].mean(dim=0)
class_p_mean = class_p_features[0:shots-1].mean(dim=0)
class_r_mean = class_r_features[0:shots-1].mean(dim=0)

features = [class_f_features, class_k_features, class_p_features, class_r_features]

c = 0
for feature in features :
    print('Class number: ' + str(c))
    for q in range(queries):
        print('Query number: ' + str(q))
        print('Distance with f centroid is: ', torch.norm(feature[shots + q]- class_f_mean, dim=-1))
        print('Distance with k centroid is: ', torch.norm(feature[shots + q]- class_k_mean, dim=-1))
        print('Distance with p centroid is: ', torch.norm(feature[shots + q]- class_p_mean, dim=-1))
        print('Distance with r centroid is: ', torch.norm(feature[shots + q]- class_r_mean, dim=-1))

    c = c + 1

feats = torch.stack((class_f_features,class_k_features, class_p_features))

print(feats.shape)

def reduce_dimension(feats):
    """
        Perfom QR decomposition on features
        - In:
            * feats : torch.Tensor with features of size [ARGS.nb_ways, ARGS.nb_shots+ARGS.nb_queries, 1024] ([3, 10, 1024])
        - Out:
            * reduced_features : torch.Tensor of dimension [ARGS.nb_ways, ARGS.nb_shots+ARGS.nb_queries, ARGS.nb_ways-1] ([3, 10 , 2])
    """
    dim = feats.shape[-1] # 1024

    # Compute the centroids of each class
    supports = feats[:, :shots]
    means = torch.mean(supports, dim=1)

    # Get Two directions of the centroids.
    perm = torch.arange(means.shape[0])-1
    directions = (means-means[perm])[:-1]

    # Get the QR decomposition from the directions of the centroids
    Q, R = torch.linalg.qr(directions.T)
    # Project Q on the data
    reduced_features = torch.matmul(feats, Q)

    return reduced_features

reduced_features = reduce_dimension(feats)
print(reduced_features.shape)


# reduced_features = reduced_features.cpu().detach().numpy()
print(reduced_features.shape)


ARGS = lambda: None
ARGS.nb_ways = 3
ARGS.nb_queries = queries
ARGS.nb_shots = shots

print(ARGS.nb_queries)

run_classes = [0,1,2]
run_indices = [list(range(20)),list(range(20)),list(range(20))]
run_classes = torch.tensor(run_classes)
run_indices = torch.tensor(run_indices)

# remove grad from reduced_features
reduced_features = reduced_features.detach()

visualize_few_shot_space(reduced_features, run_classes, run_indices, ARGS)

hover_few_shot_space(reduced_features, run_classes, run_indices, ARGS, images_path="images/")
