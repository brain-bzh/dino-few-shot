import torch
from PIL import Image
import torchvision.transforms as transforms
import os

from utils import hover_few_shot_space
import argparse

# download images from https://partage.imt.fr/index.php/s/3pJAnmkwrR7pntG
# !wget https://partage.imt.fr/index.php/s/3pJAnmkwrR7pntG/download -O images.zip
# !unzip images.zip
# download features from https://partage.imt.fr/index.php/s/rGfacYEyAK2ZXNz
# !wget https://partage.imt.fr/index.php/s/rGfacYEyAK2ZXNz/download -O features.zip
# !unzip features.zip

# parse shots and queries
parser = argparse.ArgumentParser()
parser.add_argument('--generate-features', action='store_true')
parser.add_argument('--nb-ways', type=int, default=3)
parser.add_argument('--nb-shots', type=int, default=18)
parser.add_argument('--nb-queries', type=int, default=2)
args = parser.parse_args()

assert args.nb_ways == 3, 'Only 3 ways is supported for now'
assert args.nb_shots + args.nb_queries <= 20, 'Queries + Shots should be smaller than 20 as there are only 20 images per class'

if args.generate_features:
    assert torch.cuda.is_available(), 'CUDA is not available but needed to generate features'
    assert torch.cuda.device_count() > 0, 'CUDA device is not available but needed to generate features'


inputs = []
classes_names = ['f', 'k', 'p']

# get images from each class
def get_class_images(class_name):

    transform = transforms.Compose([
    transforms.Resize(256, antialias=True),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),

    ])

    class_images = []
    for image_path in sorted(os.listdir('images/' + class_name)):
        image = Image.open('images/' + class_name + '/' + image_path)
        image = transform(image)
        class_images.append(image)
    class_images = torch.stack(class_images)
    return class_images

# load dino, generate features and save them on disk
if args.generate_features:
    dinov2_vitl14_reg = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14_reg').to('cuda')

    class_f_features = dinov2_vitl14_reg(get_class_images('f').to('cuda'))
    class_k_features = dinov2_vitl14_reg(get_class_images('k').to('cuda'))
    class_p_features = dinov2_vitl14_reg(get_class_images('p').to('cuda'))
    class_r_features = dinov2_vitl14_reg(get_class_images('r').to('cuda'))

    # save features
    torch.save(class_f_features, 'features/class_f_features.pt')
    torch.save(class_k_features, 'features/class_k_features.pt')
    torch.save(class_p_features, 'features/class_p_features.pt')
    torch.save(class_r_features, 'features/class_r_features.pt')

# load features from disk
class_f_features = torch.load('features/class_f_features.pt', map_location=torch.device('cpu'))
class_k_features = torch.load('features/class_k_features.pt', map_location=torch.device('cpu'))
class_p_features = torch.load('features/class_p_features.pt', map_location=torch.device('cpu'))

# only keep appropriate number of shots + queries
class_f_features = class_f_features[0:args.nb_shots + args.nb_queries]
class_k_features = class_k_features[0:args.nb_shots + args.nb_queries]
class_p_features = class_p_features[0:args.nb_shots + args.nb_queries]

# project features on the unit sphere and center them
def norm_center_norm(features):
    features = features / features.norm(dim=-1, keepdim=True)
    features = features - features.mean(dim=-1, keepdim=True)
    features = features / features.norm(dim=-1, keepdim=True)
    return features

class_f_features = norm_center_norm(class_f_features)
class_k_features = norm_center_norm(class_k_features)
class_p_features = norm_center_norm(class_p_features)

# nearest class mean on shots
class_f_mean = class_f_features[0:args.nb_shots].mean(dim=0)
class_k_mean = class_k_features[0:args.nb_shots].mean(dim=0)
class_p_mean = class_p_features[0:args.nb_shots].mean(dim=0)

features = [class_f_features, class_k_features, class_p_features]

# compute and display distances between queries and centroids
c = 0
for feature in features :
    print('Class: ' + classes_names[c])
    for q in range(args.nb_queries):
        distance_f = torch.norm(feature[args.nb_shots + q]- class_f_mean, dim=-1).item()
        distance_k = torch.norm(feature[args.nb_shots + q]- class_k_mean, dim=-1).item()
        distance_p = torch.norm(feature[args.nb_shots + q]- class_p_mean, dim=-1).item()
        distance_f = round(distance_f, 2)
        distance_k = round(distance_k, 2)
        distance_p = round(distance_p, 2)
        print('Distances query', q, 'from class', classes_names[c],
              'with f, k, p centroids: ', distance_f, distance_k, distance_p)
    c = c + 1

def reduce_dimension(feats):
    """
        Perfom QR decomposition on features
        - In:
            * feats : torch.Tensor with features of size [args.nb_ways, args.nb_shots+args.nb_queries, 1024] ([3, 10, 1024])
        - Out:
            * reduced_features : torch.Tensor of dimension [args.nb_ways, args.nb_shots+args.nb_queries, args.nb_ways-1] ([3, 10 , 2])
    """
    dim = feats.shape[-1] # 1024

    # compute the centroids of each class
    supports = feats[:, :args.nb_shots]
    means = torch.mean(supports, dim=1)

    # get two directions of the centroids.
    perm = torch.arange(means.shape[0])-1
    directions = (means-means[perm])[:-1]

    # get the QR decomposition from the directions of the centroids
    Q, R = torch.linalg.qr(directions.T)

    # project Q on the data
    reduced_features = torch.matmul(feats, Q)

    return reduced_features

feats = torch.stack((class_f_features,class_k_features, class_p_features))
reduced_features = reduce_dimension(feats)

# remove grad from reduced_features
reduced_features = reduced_features.detach()

hover_few_shot_space(reduced_features, args, images_path="images/")
