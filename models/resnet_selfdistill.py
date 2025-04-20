import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers.resblock import ResBlock
from .layers.bottleneck import BottleneckBlock

# ResNet class for self-distillation
class ResNetSD(nn.Module): 
    def __init__(self, block, layers, num_classes=10):
        super().__init__()

        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True) 

        # four main ResNet layers
        self.layer1 = self._make_layer(block, 64,  layers[0], stride=1, layer_index=0)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, layer_index=1)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, layer_index=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, layer_index=3)

        # final deep classifier (acts as teacher model in self-distillation)
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # self-distillation layers
        # make three shallow classifers (classifiers attached after layers 1, 2, & 3)
        self.shallow_classifiers = nn.ModuleList()
        self.bottleneck_layers = nn.ModuleList()
        feature_dims = [64 * block.expansion,
                        128 * block.expansion,
                        256 * block.expansion] # Feature dims after each layer

        # this apparently can be tuned
        bottleneck_dim = 128 

        for dim in feature_dims:
            # 1x1 conv bottleneck is added to reduce the feature dimension
            self.bottleneck_layers.append(nn.Conv2d(dim, bottleneck_dim, kernel_size=1, bias=False))

            # linear classifer added to produce logits
            self.shallow_classifiers.append(nn.Linear(bottleneck_dim, num_classes))

    def _make_layer(self, block, out_channels, num_blocks, stride, layer_index):
        layers = []
        first_stride = stride

        for i in range(num_blocks):
            stride = first_stride if i == 0 else 1
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion
            
            # use forward hook to store intermediate outputs in forward pass
            # lets us compare student features to teacher features 
            layers[-1].register_forward_hook(self._save_feature_hook(f'layer{layer_index+1}_{i}'))
        return nn.Sequential(*layers)

    def _save_feature_hook(self, name):
        def hook(module, input, output):
            if not hasattr(self, '_intermediate_features'):
                # saves intermediate feature maps in dictionary 
                self._intermediate_features = {}
            self._intermediate_features[name] = output
        return hook

    def forward(self, x):
        # reset at each forward pass
        self._intermediate_features = {} 
        out = self.relu(self.bn1(self.conv1(x))) 

        # normal ResNet forward pass thru all stages
        layer1_out = self.layer1(out)
        layer2_out = self.layer2(layer1_out)
        layer3_out = self.layer3(layer2_out)
        layer4_out = self.layer4(layer3_out)

        shallow_outputs = []
        intermediate_features_student = []
        intermediate_features_teacher = []

        # process layer1 outputs
        if len(self.bottleneck_layers) > 0:
            # get intermediate feature for the layer via the saved hook data
            # pass through bottleneck, pool it, flatten and get logits
            # save student & teacher features for loss computation in distillation_loss function
            # student features saves bottleneck outputs 
            # teacher features saves the full features before bottleneck
            feature_student = self._intermediate_features['layer1_2'] if hasattr(self.layer1, '__len__') else layer1_out
            bottleneck1 = self.bottleneck_layers[0](feature_student)
            pooled1 = self.global_avg_pool(bottleneck1)
            flattened1 = torch.flatten(pooled1, 1)
            shallow_outputs.append(self.shallow_classifiers[0](flattened1))
            intermediate_features_student.append(bottleneck1)
            
            intermediate_features_teacher.append(layer1_out)

        # process layer2 outputs
        if len(self.bottleneck_layers) > 1:
            feature_student = self._intermediate_features['layer2_3'] if hasattr(self.layer2, '__len__') else layer2_out
            bottleneck2 = self.bottleneck_layers[1](feature_student)
            pooled2 = self.global_avg_pool(bottleneck2)
            flattened2 = torch.flatten(pooled2, 1)
            shallow_outputs.append(self.shallow_classifiers[1](flattened2))
            intermediate_features_student.append(bottleneck2)

            intermediate_features_teacher.append(layer2_out)

        # process layer3 outputs
        if len(self.bottleneck_layers) > 2:
            feature_student = self._intermediate_features['layer3_5'] if hasattr(self.layer3, '__len__') else layer3_out
            bottleneck3 = self.bottleneck_layers[2](feature_student)
            pooled3 = self.global_avg_pool(bottleneck3)
            flattened3 = torch.flatten(pooled3, 1)
            shallow_outputs.append(self.shallow_classifiers[2](flattened3))
            intermediate_features_student.append(bottleneck3)

            intermediate_features_teacher.append(layer3_out)

        # final classification (deep teacher)
        pooled_final = self.global_avg_pool(layer4_out)
        flattened_final = torch.flatten(pooled_final, 1)
        final_output = F.log_softmax(self.fc(flattened_final), dim=1)
        shallow_outputs.append(final_output)
        intermediate_features_teacher.append(layer4_out) # Teacher feature at the end

        # shallow outputs has classifier outputs
        # intermediate_features_student has student features after bottlenecks
        # intermediate_features_teacher is before bottleneck
        # labels used for cross entropy loss, student vs teacher logits used for KL divergence loss, student vs teacher features used for L2 Loss
        return shallow_outputs, intermediate_features_student, intermediate_features_teacher

def resnet34_sd(num_classes=10):
    return ResNetSD(ResBlock, [3, 4, 6, 3], num_classes=num_classes)

def resnet50_sd(num_classes=10):
    return ResNetSD(BottleneckBlock, [3, 4, 6, 3], num_classes=num_classes)
