#!/usr/bin/env python3
"""
main script executing tasks defined in global settings file
"""

import os
import pickle
import numpy as np
import concurrent.futures
from sklearn.metrics import auc, roc_curve

from global_defs import CONFIG
from prepare_data import Cityscapes, Pascal_voc
from utils import vis_pred_i, compute_miou, comp_features_per_img, concat_features, plot_features, detect_adv, metrics_to_dataset, train_outlier, plot_detect_acc, init_stats, fit_model_run, stats_dump, determine_counter_attacks 

np.random.seed(0)


def main():

    """
    Load dataset
    """
    print('load dataset')

    if CONFIG.DATASET == 'cityscape':
        loader = Cityscapes( )
    elif CONFIG.DATASET == 'pascal_voc':
        loader = Pascal_voc( )
        
    print('dataset:', CONFIG.DATASET)
    print('number of images: ', len(loader))
    print('semantic segmentation network:', CONFIG.MODEL_NAME)
    print('attack:', CONFIG.ATTACK)
    print(' ')


    """
    For visualizing the (attacked) input data and predictions.
    """
    if CONFIG.PLOT_ATTACK:
        print("visualize (attacked) input data and predictions")

        if not os.path.exists( CONFIG.VIS_PRED_DIR ):
            os.makedirs( CONFIG.VIS_PRED_DIR )
        
        for i in range(len(loader)):
            vis_pred_i(loader[i])
    

    """
    Computation of mean IoU of ordinary and adversarial prediction.
    """
    if CONFIG.COMP_MIOU:
        print('compute mIoU')
        compute_miou(loader)
        compute_miou(loader, adv=True)
    

    """
    Computation of the features.
    """
    if CONFIG.COMP_FEATURES:
        print('compute features') 

        if not os.path.exists( CONFIG.COMP_FEATURES_DIR.replace(CONFIG.ATTACK+'/','') ):
            os.makedirs( CONFIG.COMP_FEATURES_DIR.replace(CONFIG.ATTACK+'/','') )
        if not os.path.exists( CONFIG.COMP_FEATURES_DIR ):
            os.makedirs( CONFIG.COMP_FEATURES_DIR )

        if CONFIG.NUM_CORES == 1:
            for i in range(len(loader)):
                comp_features_per_img(loader[i])
                comp_features_per_img(loader[i], adv=True)
        else:
            p_args = [ (loader[i],False) for i in range(len(loader)) ]
            with concurrent.futures.ProcessPoolExecutor(max_workers=CONFIG.NUM_CORES) as executor:
                executor.map(comp_features_per_img, *zip(*p_args))
            p_args = [ (loader[i],True) for i in range(len(loader)) ]
            with concurrent.futures.ProcessPoolExecutor(max_workers=CONFIG.NUM_CORES) as executor:
                executor.map(comp_features_per_img, *zip(*p_args))
        
        _, _ = concat_features()
    

    """
    For visualizing the features.
    """
    if CONFIG.VIS_VIOS:
        print('visualize features')

        if not os.path.exists( CONFIG.VIS_VIOS_DIR ):
            os.makedirs( CONFIG.VIS_VIOS_DIR )

        f_clean, f_adv = concat_features()
        plot_features(f_clean, f_adv)
    

    """
    For the detection of adversarial examples.
    """
    if CONFIG.DETECT_HISTO:
        print('detect adversarial examples')

        if not os.path.exists( CONFIG.DETECT_HISTO_DIR ):
            os.makedirs( CONFIG.DETECT_HISTO_DIR )
        
        f_clean, f_adv = concat_features()
        detect_adv(f_clean, f_adv)
    

    """
    For the detection of adversarial examples via oulier detection.
    """
    if CONFIG.DETECT_OUTLIER:
        print('detect adversarial examples via oulier detection')

        if not os.path.exists( CONFIG.DETECT_OUTLIER_DIR ):
            os.makedirs( CONFIG.DETECT_OUTLIER_DIR )
        
        f_clean, f_adv = concat_features()
        Xa, ya, X_names = metrics_to_dataset(f_clean, f_adv)

        runs = 5 # train/val splitting of 80/20
        scores_ocsvm = np.zeros((len(ya)))
        scores_ee = np.zeros((len(ya)))
        split = np.random.randint(0,runs,len(ya))   
        for i in range(runs):
            model_ocsvm, model_ee = train_outlier(Xa[split!=i,:])

            scores_ocsvm[split==i] = model_ocsvm.score_samples(Xa[split==i,:])
            scores_ee[split==i] = model_ee.score_samples(Xa[split==i,:])

        np.save(os.path.join(CONFIG.DETECT_OUTLIER_DIR, 'scores_ocsvm.npy'), scores_ocsvm)
        np.save(os.path.join(CONFIG.DETECT_OUTLIER_DIR, 'scores_ee.npy'), scores_ee)
        plot_detect_acc(f_clean, f_adv, 'ocsvm', save_path=CONFIG.DETECT_OUTLIER_DIR)
        plot_detect_acc(f_clean, f_adv, 'ee', save_path=CONFIG.DETECT_OUTLIER_DIR)
    

    """
    For the detection of adversarial examples via classification, cross val same attack.
    """
    if CONFIG.DETECT_CLASSIF:
        print('detect adversarial examples via classification (same attack)')

        if not os.path.exists( CONFIG.DETECT_CLASSIF_DIR ):
            os.makedirs( CONFIG.DETECT_CLASSIF_DIR )
        
        f_clean, f_adv = concat_features()
        Xa, ya, X_names = metrics_to_dataset(f_clean, f_adv)

        runs = 5 # train/val splitting of 80/20
        stats = init_stats(runs, X_names)
        ya_pred = np.zeros((len(ya),2))

        split = np.random.randint(0,runs,len(ya))   
        for i in range(runs):
            print('run:', i)
            stats, ya_pred_i, model = fit_model_run(Xa[split!=i,:], ya[split!=i], Xa[split==i,:], ya[split==i], stats, i)
            pickle.dump(model, open(CONFIG.DETECT_CLASSIF_DIR+'model'+str(i)+'.p', 'wb'))    
            ya_pred[split==i] = ya_pred_i
        fpr, tpr, _ = roc_curve(ya.astype(int),ya_pred[:,1])
        print('model overall auroc score:', auc(fpr, tpr) )

        stats_dump(stats, ya_pred, CONFIG.ATTACK)
        plot_detect_acc(f_clean, f_adv, CONFIG.ATTACK, save_path=CONFIG.DETECT_CLASSIF_DIR)  


    """
    For the detection of adversarial examples via classification, cross val different attack.
    """
    if CONFIG.DETECT_CROSS:
        print('detect adversarial examples via classification (different attack)')

        if not os.path.exists( CONFIG.DETECT_CROSS_DIR ):
            os.makedirs( CONFIG.DETECT_CROSS_DIR )
        
        counter_attacks, network_patch = determine_counter_attacks()

        f_clean, f_adv = concat_features()
        Xa, ya, X_names = metrics_to_dataset(f_clean, f_adv)

        for a in range(len(counter_attacks)):
            
            model_path = CONFIG.DETECT_CLASSIF_DIR.replace(CONFIG.ATTACK, counter_attacks[a])
            if counter_attacks[a] == 'patch_eot' or CONFIG.ATTACK == 'patch_eot':
                model_path = model_path.replace(CONFIG.MODEL_NAME, network_patch)
            
            if os.path.isfile(model_path+'model0.p'):
                print('Attack:', counter_attacks[a])

                runs = 5 # train/val splitting of 80/20
                ya_pred = np.zeros((len(ya),2))
                split = np.random.randint(0,runs,len(ya))   
                for i in range(runs):
            
                    model = pickle.load( open( model_path+'model'+str(i)+'.p', 'rb' ) )
                    ya_pred[split==i] = model.predict_proba(Xa[split==i,:])

                fpr, tpr, _ = roc_curve(ya.astype(int),ya_pred[:,1])
                print('model auroc score:', auc(fpr, tpr) )
                np.save(os.path.join(CONFIG.DETECT_CROSS_DIR, 'ya_pred_' + counter_attacks[a] + '.npy'), ya_pred)

                plot_detect_acc(f_clean, f_adv, counter_attacks[a], save_path=CONFIG.DETECT_CROSS_DIR)



if __name__ == '__main__':
  
    print( "===== START =====" )
    main()
    print( "===== DONE! =====" )