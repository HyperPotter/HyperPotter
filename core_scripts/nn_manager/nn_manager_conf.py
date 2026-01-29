"""
nn_manager_conf

A few definitions of nn_manager

"""
from __future__ import print_function

class CheckPointKey:
    state_dict = 'state_dict'
    info = 'info'
    optimizer = 'optimizer' 
    trnlog = 'train_log'
    vallog = 'val_log'
    lr_scheduler = 'lr_scheduler'
nn_model_keywords_default = {
    'prepare_mean_std': (True, "method to initialize mean/std"),
    'normalize_input': (True, "method to normalize input features"),
    'normalize_target': (True, "method to normalize target features"),
    'denormalize_output': (True, "method to de-normalize output features"),
    'forward': (True, "main method for forward"),
    'inference': (False, "alternative method for inference"),
    'loss': (False, 'loss defined within model module'),
    'other_setups': (False, "other setup functions before training"),
    'flag_validation': (False, 'flag to indicate train or validation set'),
    'validation': (False, 'deprecated. Please use model.flag_validation'),
    'finish_up_inference': (False, 'method to finish up work after inference')
}
nn_model_keywords_bags = {'default': nn_model_keywords_default}
loss_method_keywords_default = {
    'compute': (True, "method to comput loss")
}
loss_method_keywords_GAN = {
    'compute_gan_D_real': (True, "method to comput loss for GAN dis. on real"),
    'compute_gan_D_fake': (True, "method to comput loss for GAN dis. on fake"),
    'compute_gan_G': (True, "method to comput loss for GAN gen."),
    'compute_aux': (False, "(onlt for GAN-based model), auxialliary loss"),
    'compute_feat_match': (False, '(only for GAN-based model), feat-matching'),
    'flag_wgan': (False, '(only for GAN-based model), w-gan')
}
loss_method_keywords_bags = {'default': loss_method_keywords_default, 
                             'GAN': loss_method_keywords_GAN}

if __name__ == "__main__":
    print("Configurations for nn_manager")
