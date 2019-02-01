import os

import numpy as np
import torch
import cv2
from skimage import transform

from torch.nn import ReLU


class CamExtractor():
    """
        Extracts cam features from the model
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None

    def save_gradient(self, grad):
        self.gradients = grad

    def forward_pass_on_convolutions(self, x):
        """
            Does a forward pass on convolutions, hooks the function at given layer
        """
        conv_output = None
        for i, (module_pos, module) in enumerate(self.model.module._modules.items()):
            x = module(x)  # Forward
            if i == self.target_layer:
                x.register_hook(self.save_gradient)
                conv_output = x  # Save the convolution output on that layer
        return conv_output, x

    def forward_pass(self, x):
        """
            Does a full forward pass on the model
        """
        # Forward pass on the convolutions
        conv_output, logits_output = self.forward_pass_on_convolutions(x)
        # Forward pass on the classifier
        # x = self.model.module.logits(x) #TODO
        return conv_output, logits_output

class GradCam():
    """
        Produces class activation map
    """
    def __init__(self, model, target_layer):
        self.model = model
        # self.model.eval()
        # Define extractor
        self.extractor = CamExtractor(self.model, target_layer)

    def generate_cam(self, input_image, label_0):
        # Full forward pass
        # conv_output is the output of convolutions at specified layer
        # model_output is the final output of the model (1, 1000)
        conv_output, model_output = self.extractor.forward_pass(input_image)
        # if target_class is None:
        #     target_class = np.argmax(model_output.data.numpy())
        # # Target for backprop
        # one_hot_output = torch.FloatTensor(1, model_output.size()[-1]).zero_()
        # one_hot_output[0][target_class] = 1
        # Zero grads
        self.model.module.features.zero_grad() #TODO
        self.model.module.logits.zero_grad() #TODO
        # Backward pass with specified target, activate register_hook
        model_output.backward(gradient=label_0, retain_graph=True)
        # Get hooked gradients
        guided_gradients = self.extractor.gradients.data.cpu().numpy()[0]
        # Get convolution outputs
        target = conv_output.data.cpu().numpy()[0]
        # Get weights from gradients
        weights = np.mean(guided_gradients, axis=(1, 2))  # Take averages for each gradient
        # Create empty numpy array for cam
        cam = np.ones(target.shape[1:], dtype=np.float32)
        # Multiply each weight with its conv output and then, sum
        for i, w in enumerate(weights):
            cam += w * target[i, :, :]
        cam = cv2.resize(cam, (224, 224))
        cam = np.maximum(cam, 0)
        cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))  # Normalize between 0-1
        cam = np.uint8(cam * 255)  # Scale between 0-255 to visualize
        return cam


class GuidedBackprop():
    """
       Produces gradients generated with guided back propagation from the given image
    """
    def __init__(self, model):
        self.model = model
        self.gradients = None
        # Put model in evaluation mode
        # self.model.eval()
        self.update_relus()
        self.hook_layers()

    def hook_layers(self):
        def hook_function(module, grad_in, grad_out):
            self.gradients = grad_in[0]

        # Register hook to the first layer
        first_layer = list(self.model.module.features._modules.items())[0][1]
        first_layer.register_backward_hook(hook_function)

    def update_relus(self):
        """
            Updates relu activation functions so that it only returns positive gradients
        """
        def relu_hook_function(module, grad_in, grad_out):
            """
            If there is a negative gradient, changes it to zero
            """
            if isinstance(module, ReLU):
                return (torch.clamp(grad_in[0], min=0.0),)
        # Loop through layers, hook up ReLUs with relu_hook_function
        for pos, module in self.model.module.features._modules.items():
            if isinstance(module, ReLU):
                module.register_backward_hook(relu_hook_function)

    def generate_gradients(self, image, label_0):
        # Forward pass
        model_output = self.model(image)
        # Zero gradients
        self.model.zero_grad()
        # Target for backprop
        # Backward pass
        model_output.backward(gradient=label_0)
        # Convert Pytorch variable to numpy array
        # [0] to get rid of the first channel (1,3,224,224)
        gradients_as_arr = self.gradients.data.cpu().numpy()[0]
        return gradients_as_arr

def guided_grad_cam(grad_cam_mask, guided_backprop_mask):
    """
        Guided grad cam is just pointwise multiplication of cam mask and
        guided backprop mask

    Args:
        grad_cam_mask (np_arr): Class activation map mask
        guided_backprop_mask (np_arr):Guided backprop mask
    """
    # ValueError: operands could not be broadcast together with shapes (224,224) (32,111,111)
    guided_backprop_mask = np.mean(guided_backprop_mask, axis=0)
    # print("guided_backprop_mask.shape = {}".format(guided_backprop_mask.shape))
    guided_backprop_mask = transform.resize(guided_backprop_mask, (224, 224))
    # print("guided_backprop_mask.shape = {}".format(guided_backprop_mask.shape))
    cam_gb = np.multiply(grad_cam_mask, guided_backprop_mask)
    # print("cam_gbshape = {}".format(cam_gb.shape))
    cam_gb = np.array([cam_gb, cam_gb, cam_gb])
    return cam_gb


def save_gradient_images(gradient, file_name):
    """
        Exports the original gradient image

    Args:
        gradient (np arr): Numpy array of the gradient with shape (3, 224, 224)
        file_name (str): File name to be exported
    """
    gradient = gradient - gradient.min()
    gradient /= gradient.max()
    gradient = np.uint8(gradient * 255).transpose(1, 2, 0)
    path_to_file = os.path.join(file_name)
    # Convert RBG to GBR
    gradient = gradient[..., ::-1]
    cv2.imwrite(path_to_file, gradient)

def convert_to_grayscale(cv2im):
    """
        Converts 3d image to grayscale

    Args:
        cv2im (numpy arr): RGB image with shape (D,W,H)

    returns:
        grayscale_im (numpy_arr): Grayscale image with shape (1,W,D)
    """
    grayscale_im = np.sum(np.abs(cv2im), axis=0)
    im_max = np.percentile(grayscale_im, 99)
    im_min = np.min(grayscale_im)
    grayscale_im = (np.clip((grayscale_im - im_min) / (im_max - im_min), 0, 1))
    return grayscale_im

def cam(net, image, labels_0, target_layer=0):
    print("Set Model Trainning mode to trainning=[{}]".format(net.eval().training))
    gcv2 = GradCam(net, target_layer) # usually last conv layer
    # Generate cam mask
    cam = gcv2.generate_cam(image, labels_0)

    # Guided backprop
    GBP = GuidedBackprop(net)
    # Get gradients
    guided_grads = GBP.generate_gradients(image, labels_0)

    # Guided Grad cam
    cam_gb = guided_grad_cam(cam, guided_grads)
    # save_gradient_images(cam_gb, config.DIRECTORY_CSV+"_img.jpg")
    return convert_to_grayscale(cam_gb)
    # grayscale_cam_gb = np.expand_dims(grayscale_cam_im, axis=0)
    # save_gradient_images(grayscale_cam_gb, config.DIRECTORY_CSV + '_img_gray.jpg')