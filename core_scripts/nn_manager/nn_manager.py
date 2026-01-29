"""
nn_manager for GAN

A simple wrapper to run the training / testing process

"""
from __future__ import print_function

import time
import datetime
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import core_scripts.data_io.conf as nii_dconf
import core_scripts.other_tools.display as nii_display
import core_scripts.other_tools.str_tools as nii_str_tk
import core_scripts.op_manager.op_process_monitor as nii_monitor
import core_scripts.op_manager.op_display_tools as nii_op_display_tk
import core_scripts.nn_manager.nn_manager_tools as nii_nn_tools
import core_scripts.nn_manager.nn_manager_conf as nii_nn_manage_conf


def f_run_one_epoch(args,
                    pt_model, loss_wrapper, \
                    device, monitor,  \
                    data_loader, epoch_idx, optimizer = None, \
                    target_norm_method = None):
    """
    f_run_one_epoch: 
       run one poech over the dataset (for training or validation sets)

    Args:
       args:         from argpase
       pt_model:     pytorch model (torch.nn.Module)
       loss_wrapper: a wrapper over loss function
                     loss_wrapper.compute(generated, target) 
       device:       torch.device("cuda") or torch.device("cpu")
       monitor:      defined in op_procfess_monitor.py
       data_loader:  pytorch DataLoader. 
       epoch_idx:    int, index of the current epoch
       optimizer:    torch optimizer or None
                     if None, the back propgation will be skipped
                     (for developlement set)
       target_norm_method: method to normalize target data
                           (by default, use pt_model.normalize_target)
    """
    start_time = time.time()
    for data_idx, (data_in, data_tar, data_info, idx_orig) in \
        enumerate(data_loader):
        if optimizer is not None:
            optimizer.zero_grad()
        data_in = data_in.to(device, dtype=nii_dconf.d_dtype)
        if args.model_forward_with_target:
            if isinstance(data_tar, torch.Tensor):
                data_tar_tm = data_tar.to(device, dtype=nii_dconf.d_dtype)
                if args.model_forward_with_file_name:
                    data_gen = pt_model(data_in, data_tar_tm, data_info)
                else:
                    data_gen = pt_model(data_in, data_tar_tm)
            else:
                nii_display.f_print("--model-forward-with-target is set")
                nii_display.f_die("but data_tar is not loaded")
        else:
            if args.model_forward_with_file_name:
                data_gen = pt_model(data_in, data_info)
            else:
                data_gen = pt_model(data_in)

        if hasattr(pt_model, 'loss'):
            if isinstance(data_tar, torch.Tensor):
                data_tar = data_tar.to(device, dtype=nii_dconf.d_dtype)
            else:
                data_tar = []
            
            loss_computed = pt_model.loss(data_gen, data_tar)
        else:
            if isinstance(data_tar, torch.Tensor):
                data_tar = data_tar.to(device, dtype=nii_dconf.d_dtype)
                if target_norm_method is None:
                    normed_target = pt_model.normalize_target(data_tar)
                else:
                    normed_target = target_norm_method(data_tar)
            else:
                normed_target = []
            loss_computed = loss_wrapper.compute(data_gen, normed_target)

        loss_values = [0]
        loss, loss_values, loss_flags = nii_nn_tools.f_process_loss(
            loss_computed)
        if optimizer is not None:
            loss.backward()
            if args.grad_clip_norm > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    pt_model.parameters(), args.grad_clip_norm)
            optimizer.step()
        end_time = time.time()
        batchsize = len(data_info)
        for idx, data_seq_info in enumerate(data_info):
            monitor.log_loss(loss_values, loss_flags, \
                             (end_time-start_time) / batchsize, \
                             data_seq_info, idx_orig.numpy()[idx], \
                             epoch_idx)
            if args.verbose == 1:
                monitor.print_error_for_batch(data_idx*batchsize + idx,\
                                              idx_orig.numpy()[idx], \
                                              epoch_idx)
        start_time = time.time()
    return
    

def f_train_wrapper(args, pt_model, loss_wrapper, device, \
                    optimizer_wrapper, \
                    train_dataset_wrapper, \
                    val_dataset_wrapper = None, \
                    checkpoint = None):
    """ 
    f_train_wrapper(args, pt_model, loss_wrapper, device, 
                    optimizer_wrapper
                    train_dataset_wrapper, val_dataset_wrapper = None,
                    check_point = None):
      A wrapper to run the training process

    Args:
       args:         argument information given by argpase
       pt_model:     pytorch model (torch.nn.Module)
       loss_wrapper: a wrapper over loss function
                     loss_wrapper.compute(generated, target) 
       device:       torch.device("cuda") or torch.device("cpu")

       optimizer_wrapper: 
           a wrapper over optimizer (defined in op_manager.py)
           optimizer_wrapper.optimizer is torch.optimizer
    
       train_dataset_wrapper: 
           a wrapper over training data set (data_io/default_data_io.py)
           train_dataset_wrapper.get_loader() returns torch.DataSetLoader
       
       val_dataset_wrapper: 
           a wrapper over validation data set (data_io/default_data_io.py)
           it can None.
       
       check_point:
           a check_point that stores every thing to resume training
    """        
    
    nii_display.f_print_w_date("Start model training")
    optimizer_wrapper.print_info()
    optimizer = optimizer_wrapper.optimizer
    lr_scheduler = optimizer_wrapper.lr_scheduler
    epoch_num = optimizer_wrapper.get_epoch_num()
    no_best_epoch_num = optimizer_wrapper.get_no_best_epoch_num()
    train_dataset_wrapper.print_info()
    train_data_loader = train_dataset_wrapper.get_loader()
    train_seq_num = train_dataset_wrapper.get_seq_num()
    monitor_trn = nii_monitor.Monitor(epoch_num, train_seq_num)
    if val_dataset_wrapper is not None:
        val_dataset_wrapper.print_info()
        val_data_loader = val_dataset_wrapper.get_loader()
        val_seq_num = val_dataset_wrapper.get_seq_num()
        monitor_val = nii_monitor.Monitor(epoch_num, val_seq_num)
    else:
        monitor_val = None
    train_log = ''
    if torch.cuda.device_count() > 1 and args.multi_gpu_data_parallel:
        flag_multi_device = True  
        nii_display.f_print("\nUse %d GPUs\n" % (torch.cuda.device_count()))
        normtarget_f = pt_model.normalize_target
        pt_model = nn.DataParallel(pt_model)
    else:
        nii_display.f_print("\nUse single GPU: %s\n" % \
                            (torch.cuda.get_device_name(device)))
        flag_multi_device = False
        normtarget_f = None
    pt_model.to(device, dtype=nii_dconf.d_dtype)
    nii_nn_tools.f_model_show(pt_model)
    nii_nn_tools.f_loss_show(loss_wrapper)
    cp_names = nii_nn_manage_conf.CheckPointKey()
    if checkpoint is not None:
        if type(checkpoint) is dict:
            if cp_names.state_dict in checkpoint:
                pt_model.load_state_dict(
                    nii_nn_tools.f_state_dict_wrapper(
                        checkpoint[cp_names.state_dict], 
                        flag_multi_device))
            if cp_names.optimizer in checkpoint and \
               not args.ignore_optimizer_statistics_in_trained_model:
                optimizer.load_state_dict(checkpoint[cp_names.optimizer])
            if not args.ignore_training_history_in_trained_model:
                if cp_names.trnlog in checkpoint:
                    monitor_trn.load_state_dic(
                        checkpoint[cp_names.trnlog])
                if cp_names.vallog in checkpoint and monitor_val:
                    monitor_val.load_state_dic(
                        checkpoint[cp_names.vallog])
                if cp_names.info in checkpoint:
                    train_log = checkpoint[cp_names.info]
                if cp_names.lr_scheduler in checkpoint and \
                   checkpoint[cp_names.lr_scheduler] and lr_scheduler.f_valid():
                    lr_scheduler.f_load_state_dict(
                        checkpoint[cp_names.lr_scheduler])
                    
                nii_display.f_print("Load check point, resume training")
            else:
                nii_display.f_print("Load pretrained model and optimizer")
        else:
            pt_model.load_state_dict(
                nii_nn_tools.f_state_dict_wrapper(
                    checkpoint, flag_multi_device))
            nii_display.f_print("Load pretrained model")
    if hasattr(pt_model, "other_setups"):
        nii_display.f_print("Conduct User-defined setup")
        pt_model.other_setups()
    if hasattr(pt_model, "g_pretrained_model_path") and \
       hasattr(pt_model, "g_pretrained_model_prefix"):
        nii_display.f_print("Load pret-rained models as part of this mode")
        nii_nn_tools.f_load_pretrained_model_partially(
            pt_model, pt_model.g_pretrained_model_path, 
            pt_model.g_pretrained_model_prefix)
    flag_early_stopped = False
    start_epoch = monitor_trn.get_epoch()
    epoch_num = monitor_trn.get_max_epoch()
    _ = nii_op_display_tk.print_log_head()
    nii_display.f_print_message(train_log, flush=True, end='')
    for epoch_idx in range(start_epoch, epoch_num):
        pt_model.train()
        if hasattr(pt_model, 'validation'):
            pt_model.validation = False
            mes = "Warning: model.validation is deprecated, "
            mes += "please use model.flag_validation"
            nii_display.f_print(mes, 'warning')
        if hasattr(pt_model, 'flag_validation'):
            pt_model.flag_validation = False

        f_run_one_epoch(args, pt_model, loss_wrapper, device, \
                        monitor_trn, train_data_loader, \
                        epoch_idx, optimizer, normtarget_f)
        time_trn = monitor_trn.get_time(epoch_idx)
        loss_trn = monitor_trn.get_loss(epoch_idx)
        if val_dataset_wrapper is not None:
            if args.eval_mode_for_validation:
                pt_model.eval()
            if hasattr(pt_model, 'validation'):
                pt_model.validation = True
                mes = "Warning: model.validation is deprecated, "
                mes += "please use model.flag_validation"
                nii_display.f_print(mes, 'warning')
            if hasattr(pt_model, 'flag_validation'):
                pt_model.flag_validation = True

            with torch.no_grad():
                f_run_one_epoch(args, pt_model, loss_wrapper, \
                                device, \
                                monitor_val, val_data_loader, \
                                epoch_idx, None, normtarget_f)
            time_val = monitor_val.get_time(epoch_idx)
            loss_val = monitor_val.get_loss(epoch_idx)
            if lr_scheduler.f_valid():
                lr_scheduler.f_step(loss_val)

        else:
            time_val, loss_val = 0, 0
                
        
        if val_dataset_wrapper is not None:
            flag_new_best = monitor_val.is_new_best()
        else:
            flag_new_best = True
        train_log += nii_op_display_tk.print_train_info(
            epoch_idx, time_trn, loss_trn, time_val, loss_val, 
            flag_new_best, optimizer_wrapper.get_lr_info())
        if flag_new_best:
            tmp_best_name = nii_nn_tools.f_save_trained_name(args)
            torch.save(pt_model.state_dict(), tmp_best_name)
        if not args.not_save_each_epoch:
            tmp_model_name = nii_nn_tools.f_save_epoch_name(args, epoch_idx)
            
            if monitor_val is not None:
                tmp_val_log = monitor_val.get_state_dic()
            else:
                tmp_val_log = None
                
            if lr_scheduler.f_valid():
                lr_scheduler_state = lr_scheduler.f_state_dict()
            else:
                lr_scheduler_state = None
            tmp_dic = {
                cp_names.state_dict : pt_model.state_dict(),
                cp_names.info : train_log,
                cp_names.optimizer : optimizer.state_dict(),
                cp_names.trnlog : monitor_trn.get_state_dic(),
                cp_names.vallog : tmp_val_log,
                cp_names.lr_scheduler : lr_scheduler_state
            }
            torch.save(tmp_dic, tmp_model_name)
            if args.verbose == 1:
                nii_display.f_eprint(str(datetime.datetime.now()))
                nii_display.f_eprint("Save {:s}".format(tmp_model_name),
                                     flush=True)
        if lr_scheduler.f_allow_early_stopping() and \
           monitor_val is not None and \
           monitor_val.should_early_stop(no_best_epoch_num):
            flag_early_stopped = True
            break
    nii_op_display_tk.print_log_tail()
    if flag_early_stopped:
        nii_display.f_print("Training finished by early stopping")
    else:
        nii_display.f_print("Training finished")
    nii_display.f_print("Model is saved to", end = '')
    nii_display.f_print("{}".format(nii_nn_tools.f_save_trained_name(args)))
    return


def f_inference_wrapper(args, pt_model, device, \
                        test_dataset_wrapper, checkpoint):
    """ Wrapper for inference
    """
    test_data_loader = test_dataset_wrapper.get_loader()
    test_seq_num = test_dataset_wrapper.get_seq_num()
    test_dataset_wrapper.print_info()
    if torch.cuda.device_count() > 1 and args.multi_gpu_data_parallel:
        nii_display.f_print(
            "DataParallel for inference is not implemented", 'warning')
    nii_display.f_print("\nUse single GPU: %s\n" % \
                        (torch.cuda.get_device_name(device)))
    pt_model.to(device, dtype=nii_dconf.d_dtype)
    nii_nn_tools.f_model_show(pt_model)
    cp_names = nii_nn_manage_conf.CheckPointKey()
    if type(checkpoint) is dict and cp_names.state_dict in checkpoint:
        pt_model.load_state_dict(checkpoint[cp_names.state_dict])
    else:
        pt_model.load_state_dict(checkpoint)
    nii_display.f_print("Start inference (generation):", 'highlight')
    
    pt_model.eval() 
    with torch.no_grad():
        for _, (data_in, data_tar, data_info, idx_orig) in \
            enumerate(test_data_loader):
            data_in = data_in.to(device, dtype=nii_dconf.d_dtype)
            if isinstance(data_tar, torch.Tensor):
                data_tar = data_tar.to(device, dtype=nii_dconf.d_dtype)
            start_time = time.time()
            if hasattr(pt_model, "inference"):
                infer_func = pt_model.inference
            else:
                infer_func = pt_model.forward

            if args.model_forward_with_target:
                if args.model_forward_with_file_name:
                    data_gen = infer_func(data_in, data_tar, data_info)
                else:
                    data_gen = infer_func(data_in, data_tar)
            else:    
                if args.model_forward_with_file_name:
                    data_gen = infer_func(data_in, data_info)
                else:
                    data_gen = infer_func(data_in)
            
            time_cost = time.time() - start_time
            time_cost = time_cost / len(data_info)
                
            if data_gen is None:
                nii_display.f_print("No output saved: %s" % (str(data_info)),\
                                    'warning')
                for idx, seq_info in enumerate(data_info):
                    _ = nii_op_display_tk.print_gen_info(seq_info, time_cost)
                continue
            else:
                try:
                    data_gen = pt_model.denormalize_output(data_gen)
                    data_gen_np = data_gen.to("cpu").numpy()
                except AttributeError:
                    mes = "Output data is not torch.tensor. Please check "
                    mes += "model.forward or model.inference"
                    nii_display.f_die(mes)
                for idx, seq_info in enumerate(data_info):
                    _ = nii_op_display_tk.print_gen_info(seq_info, time_cost)
                    test_dataset_wrapper.putitem(data_gen_np[idx:idx+1],\
                                                 args.output_dir, \
                                                 seq_info)
    nii_display.f_print("Generated data to %s" % (args.output_dir))
    if hasattr(pt_model, "finish_up_inference"):
        pt_model.finish_up_inference()
    return
            
if __name__ == "__main__":
    print("nn_manager")
