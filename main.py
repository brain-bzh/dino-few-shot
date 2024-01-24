import torch
from PIL import Image
import torchvision.transforms as transforms
import os
import matplotlib
import matplotlib.pyplot as plt
from utils import visualize_few_shot_space
from utils import hover_few_shot_space

ways = 2
shots = 8
queries = 2

torch.set_default_device('cuda')

dinov2_vitl14_reg = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14_reg')

inputs = []

transform = transforms.Compose([
    transforms.Resize(256, antialias=True),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])



i = 0
inputs = []

for images in sorted(os.listdir('images')):
    print(images)
    image = Image.open('images/'+images)

    img_tensor = transform(image)
    inputs.append(img_tensor)

inputs = torch.stack(inputs).to('cuda')

features = dinov2_vitl14_reg(inputs)

def norm_center_norm(features):
    # L2 normalization
    features = features / features.norm(dim=-1, keepdim=True)
    # centering
    features = features - features.mean(dim=-1, keepdim=True)
    # L2 normalization
    features = features / features.norm(dim=-1, keepdim=True)
    return features

features = norm_center_norm(features)

# Nearest Class Mean
class_1 = features[0:7].mean(dim=0)
class_2 = features[10:17].mean(dim=0)
class_3 = features[20:27].mean(dim=0)

diff81 = torch.norm(features[8] - class_1,dim=-1)
diff82 = torch.norm(features[8] - class_2,dim=-1)
diff83 = torch.norm(features[8] - class_3,dim=-1)
diff91 = torch.norm(features[9] - class_1,dim=-1)
diff92 = torch.norm(features[9] - class_2,dim=-1)
diff93 = torch.norm(features[9] - class_3,dim=-1)

diff181 = torch.norm(features[18] - class_1,dim=-1)
diff182 = torch.norm(features[18] - class_2,dim=-1)
diff183 = torch.norm(features[18] - class_3,dim=-1)
diff191 = torch.norm(features[19] - class_1,dim=-1)
diff192 = torch.norm(features[19] - class_2,dim=-1)
diff193 = torch.norm(features[19] - class_3,dim=-1)

diff281 = torch.norm(features[28] - class_1,dim=-1)
diff282 = torch.norm(features[28] - class_2,dim=-1)
diff283 = torch.norm(features[28] - class_3,dim=-1)
diff291 = torch.norm(features[29] - class_1,dim=-1)
diff292 = torch.norm(features[29] - class_2,dim=-1)
diff293 = torch.norm(features[29] - class_3,dim=-1)



print('diff: ', diff81)
print('diff: ', diff82)
print('diff: ', diff83)
print('diff: ', diff91)
print('diff: ', diff92)
print('diff: ', diff93)

print('\n')

print('diff: ', diff181)
print('diff: ', diff182)
print('diff: ', diff183)
print('diff: ', diff191)
print('diff: ', diff192)
print('diff: ', diff193)

print('\n')


print('diff: ', diff281)
print('diff: ', diff282)
print('diff: ', diff283)
print('diff: ', diff291)
print('diff: ', diff292)
print('diff: ', diff293)



feats_1 = features[0:10]
feats_2 = features[10:20]
feats_3 = features[20:30]

feats = torch.stack((feats_1,feats_2, feats_3))

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
    supports = feats[:, :shots] # TO DO
    means = torch.mean(supports, dim=1) # TO DO

    # Get Two directions of the centroids.
    perm = torch.arange(means.shape[0])-1 # TO DO
    directions = (means-means[perm])[:-1] # TO DO

    # Get the QR decomposition from the directions of the centroids
    Q, R = torch.linalg.qr(directions.T)
    # Project Q on the data
    reduced_features = torch.matmul(feats, Q) # TO DO

    return reduced_features

reduced_features = reduce_dimension(feats)
print(reduced_features.shape)


# reduced_features = reduced_features.cpu().detach().numpy()
print(reduced_features.shape)


ARGS = lambda: None
ARGS.nb_ways = 3
ARGS.nb_queries = 2
ARGS.nb_shots = 8

print(ARGS.nb_queries)

run_classes = [0,1,2]
run_indices = [[0,1,2,3,4,5,6,7,8,9],[0,1,2,3,4,5,6,7,8,9],[0,1,2,3,4,5,6,7,8,9]]
run_classes = torch.tensor(run_classes)
run_indices = torch.tensor(run_indices)

# remove grad from reduced_features
reduced_features = reduced_features.detach()

visualize_few_shot_space(reduced_features, run_classes, run_indices, ARGS)

hover_few_shot_space(reduced_features, run_classes, run_indices, ARGS, images_path="images/")
