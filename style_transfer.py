import torch
from torch import nn
from torchvision.models import vgg19
import torchvision.transforms as transforms


total_step = 200
content_layers = [7]
content_layer_weights = [1]
style_layers = [0, 2, 5, 7, 10]
style_layer_weights = [0.2, 0.2, 0.2, 0.2, 0.2]
content_weight = 1
style_weight = 100000000

img_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

assert len(content_layers) == len(content_layer_weights)
assert len(style_layers) == len(style_layer_weights)
assert sum(content_layer_weights) == sum(style_layer_weights) == 1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = vgg19(pretrained=True).features.to(device).eval()
print(model)


def get_feature_map(img, layer):
    new_model = nn.Sequential(*list(model.children())[0:layer])
    feature_map = new_model(img)
    return feature_map / 2


def get_gram_matrix(img, layer):
    feature_map = get_feature_map(img, layer)
    n, c, h, w = feature_map.size()
    assert(n == 1)
    feature_map = feature_map.view(c, h * w)
    gram_matrix = torch.mm(feature_map, feature_map.t())
    normalized_gram_matrix = gram_matrix / (n * c * h * w)
    return normalized_gram_matrix


class ContentLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, source, target):
        loss = 0
        for i in range(len(content_layers)):
            layer = content_layers[i]
            weight = content_layer_weights[i]
            source_feature_map = get_feature_map(source, layer)
            target_feature_map = get_feature_map(target, layer)
            loss += weight * self.mse(source_feature_map, target_feature_map)
        return loss


class StyleLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, source, target):
        loss = 0
        for i in range(len(style_layers)):
            layer = style_layers[i]
            weight = style_layer_weights[i]
            source_gram_matrix = get_gram_matrix(source, layer)
            target_gram_matrix = get_gram_matrix(target, layer)
            loss += weight * self.mse(source_gram_matrix, target_gram_matrix)
        return loss


content_criterion = ContentLoss()
style_criterion = StyleLoss()


def get_loss(input_img, content_img, style_img):
    content_loss = content_criterion(input_img, content_img)
    style_loss = style_criterion(input_img, style_img)
    return content_weight * content_loss + style_weight * style_loss


def style_transfer(content_img, style_img, input_img=None):
    content_img = img_transform(content_img).unsqueeze(0).to(device)
    style_img = img_transform(style_img).unsqueeze(0).to(device)
    if input_img is None:
        input_img = torch.rand(content_img.size()).to(device)
    else:
        input_img = img_transform(input_img).unsqueeze(0).to(device)
    input_img = input_img.requires_grad_()

    optimizer = torch.optim.Adam([input_img], lr=1e-2)
    for step in range(total_step):
        loss = get_loss(input_img, content_img, style_img)
        print("%s: %s" % (step, loss.item()))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    input_img.data.clamp_(0, 1)
    return input_img
