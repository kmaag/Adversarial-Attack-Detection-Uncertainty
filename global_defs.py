#!/usr/bin/env python3
'''
script including
class object with global settings
'''

class CONFIG:
  
    #---------------------#
    # set necessary paths #
    #---------------------#
  
    io_path   = '/outputs/'   # directory saving outputs

    #------------------#
    # select or define #
    #------------------#
  
    datasets = ['cityscape', 'pascal_voc']
    DATASET = datasets[0]

    model_names = ['DeepLabV3_Plus_xception65','HRNet_hrnet_w18_small_v1','ddrnet23Slim','bisenetX39']
    MODEL_NAME = model_names[0]
    
    FGSM_eps = 4 # 4,8,16
    attacks = ['FGSM_untargeted'+str(FGSM_eps),'FGSM_targeted'+str(FGSM_eps),'FGSM_untargeted_iterative'+str(FGSM_eps),'FGSM_targeted_iterative'+str(FGSM_eps),'smm_static','smm_dynamic','patch_eot']
    ATTACK = attacks[0]

    #----------------------------#
    # paths for data preparation #
    #----------------------------#
    
    DATA_DIR   = '/home/data/'
    PROBS_DIR  = '/home/probs_clean/' 
    PROBSA_DIR = '/home/probs_perturbed/' 

    #--------------------------------------------------------------------#
    # select tasks to be executed by setting boolean variable True/False #
    #--------------------------------------------------------------------#

    PLOT_ATTACK    = False
    COMP_MIOU      = False 
    COMP_FEATURES  = False 
    VIS_VIOS       = False
    DETECT_HISTO   = False 
    DETECT_OUTLIER = False
    DETECT_CLASSIF = False 
    DETECT_CROSS   = False 
    

    #-----------#
    # optionals #
    #-----------#

    NUM_CORES = 1
    
    SAVE_OUT_DIR       = io_path + DATASET + '/' + MODEL_NAME + '/' 
    VIS_PRED_DIR       = io_path + DATASET + '/' + MODEL_NAME + '/' + ATTACK + '/vis_pred/'
    COMP_MIOU_DIR      = io_path + DATASET + '/' + MODEL_NAME + '/' + ATTACK + '/miou/'
    COMP_FEATURES_DIR  = io_path + DATASET + '/' + MODEL_NAME + '/' + ATTACK + '/features/'
    VIS_VIOS_DIR       = io_path + DATASET + '/' + MODEL_NAME + '/' + ATTACK + '/vis_vios/'
    DETECT_HISTO_DIR   = io_path + DATASET + '/' + MODEL_NAME + '/' + ATTACK + '/detect_histo/'
    DETECT_OUTLIER_DIR = io_path + DATASET + '/' + MODEL_NAME + '/' + ATTACK + '/detect_outlier/'
    DETECT_CLASSIF_DIR = io_path + DATASET + '/' + MODEL_NAME + '/' + ATTACK + '/detect_classif/'
    DETECT_CROSS_DIR   = io_path + DATASET + '/' + MODEL_NAME + '/' + ATTACK + '/detect_cross/'
    
    