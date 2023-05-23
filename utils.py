#!/usr/bin/env python3
'''
script including utility functions
'''

import os
import sys
import pickle
import numpy as np 
import pandas as pd
from PIL import Image
import seaborn as sns
from scipy.stats import entropy
from sklearn.metrics import auc, roc_curve
from sklearn import linear_model
from sklearn.svm import OneClassSVM
from sklearn.covariance import EllipticEnvelope
import matplotlib.pyplot as plt

from global_defs import CONFIG
from prepare_data import cs_labels, voc_labels

plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"


def vis_pred_i(item):

    if CONFIG.DATASET == 'cityscape':
        labels = cs_labels
        classes = np.concatenate( (np.arange(0,19),[255]), axis=0)
    elif CONFIG.DATASET == 'pascal_voc':
        labels = voc_labels
        classes = np.concatenate( (np.arange(0,21),[255]), axis=0)
    trainId2label = { label.trainId : label for label in reversed(labels) }

    image = item[0]
    label = item[1]
    probs = np.load(item[3])
    seg = np.argmax(probs, axis=0)
    probs_adv = np.load(item[4])
    seg_adv = np.argmax(probs_adv, axis=0)

    I1 = image.copy()
    I2 = image.copy()
    I3 = image.copy()
    I4 = image.copy()

    for c in classes:
        I2[label==c,:] = np.asarray(trainId2label[c].color)
        I3[seg==c,:] = np.asarray(trainId2label[c].color)
        I4[seg_adv==c,:] = np.asarray(trainId2label[c].color)
    
    plt.imsave(CONFIG.VIS_PRED_DIR + item[2] + '_tmp1.png', entropy(probs,axis=0), cmap='inferno')
    I5 = np.asarray( Image.open(CONFIG.VIS_PRED_DIR + item[2] + '_tmp1.png').convert('RGB') )
    os.remove(CONFIG.VIS_PRED_DIR + item[2] + '_tmp1.png')

    plt.imsave(CONFIG.VIS_PRED_DIR + item[2] + '_tmp2.png', entropy(probs_adv,axis=0), cmap='inferno')
    I6 = np.asarray( Image.open(CONFIG.VIS_PRED_DIR + item[2] + '_tmp2.png').convert('RGB') )
    os.remove(CONFIG.VIS_PRED_DIR + item[2] + '_tmp2.png')

    img12   = np.concatenate( (I2,I1), axis=0 )
    img34  = np.concatenate( (I3,I4), axis=1 )
    img56  = np.concatenate( (I5,I6), axis=1 )
    img3456   = np.concatenate( (img34,img56), axis=0 )
    img   = np.concatenate( (img12,img3456), axis=1 )

    img = Image.fromarray(img.astype('uint8'), 'RGB')
    img.save(CONFIG.VIS_PRED_DIR + item[2] + '.png')
    plt.close()
    print('stored:', item[2]+'.png')


def compute_miou(loader, adv=False):

    if adv:
        save_path = os.path.join(CONFIG.COMP_MIOU_DIR,'miou_acc_apsr.npy')
    else:
        save_path = os.path.join(CONFIG.COMP_MIOU_DIR.replace(CONFIG.ATTACK+'/',''),'miou_acc_apsr.npy')
    
    if not os.path.exists( os.path.dirname(save_path) ):
        os.makedirs( os.path.dirname(save_path) )

    if not os.path.isfile(save_path):

        if CONFIG.DATASET == 'cityscape':
            num_classes = 19
            class_id = 11
            labels = cs_labels
        elif CONFIG.DATASET == 'pascal_voc':
            num_classes = 21
            class_id = 15
            labels = voc_labels
        trainId2label = { label.trainId : label for label in reversed(labels) }

        seg_all = []
        gt_all = []

        for item in loader:
            if CONFIG.ATTACK != 'smm_dynamic' or class_id in np.unique(item[1]):
                print(item[2])

                gt = item[1]
                if adv:
                    seg = np.argmax(np.load(item[4]), axis=0)
                else:
                    seg = np.argmax(np.load(item[3]), axis=0)
                seg[gt==255] = 255

                seg_all.append(seg)
                gt_all.append(gt)
        
        seg_all = np.stack(seg_all, 0)
        gt_all = np.stack(gt_all, 0)

        seg_iu = np.zeros((num_classes,2))
        num_pix = 0

        for c in range(num_classes):
            seg_iu[c,0] = np.sum(np.logical_and(seg_all==c,gt_all==c))
            seg_iu[c,1] = np.sum(np.logical_or(seg_all==c,gt_all==c))
        num_pix = np.sum(gt_all != 255)

        result_path = save_path.replace('npy','txt')
        with open(result_path, 'a') as fi:
            print('(adversarial) prediction ',  adv, ':', file=fi)
            counter_c = 0
            iou_all = 0
            for c in range(num_classes):
                if seg_iu[c,1] > 0:
                    counter_c += 1
                    iou_c = seg_iu[c,0] / seg_iu[c,1]
                    iou_all += iou_c
                    print('IoU of class', trainId2label[c].name, ':', iou_c, file=fi)
            print('mIoU:', iou_all / counter_c, file=fi)
            print('accuracy:', np.sum(seg_iu[:,0]) / num_pix, file=fi)
            print('APSR:', (num_pix-np.sum(seg_iu[:,0])) / num_pix, file=fi)
            print(' ', file=fi)
        
        metrics = np.zeros((3))
        metrics[0] = iou_all / counter_c
        metrics[1] = np.sum(seg_iu[:,0]) / num_pix
        metrics[2] = (num_pix-np.sum(seg_iu[:,0])) / num_pix
        np.save(save_path, metrics)


def variation_ratio( probs ):
    output = np.ones((probs.shape[1],probs.shape[2]))
    return output - np.sort(probs, axis=0)[-1,:,:]
  

def probdist( probs ):
    output = np.ones((probs.shape[1],probs.shape[2]))
    return output - np.sort(probs, axis=0)[-1,:,:] + np.sort(probs, axis=0)[-2,:,:]


def comp_features_per_img(item, adv=False):

    if CONFIG.DATASET == 'cityscape':
        num_classes = 19
        class_id = 11
    elif CONFIG.DATASET == 'pascal_voc':
        num_classes = 21
        class_id = 15

    if adv:
        save_path = CONFIG.COMP_FEATURES_DIR + 'features_' + item[2] + '_' + str(adv) + '.p'
    else:
        save_path = CONFIG.COMP_FEATURES_DIR.replace(CONFIG.ATTACK+'/','') + 'features_' + item[2] + '_' + str(adv) + '.p'
    
    if not os.path.isfile(save_path) and (CONFIG.ATTACK != 'smm_dynamic' or class_id in np.unique(item[1])):

        print(item[2])

        features = { 'name': list([]), 'miou': list([]), 'acc': list([]), 'apsr': list([]), 'mE': list([]), 'mV': list([]), 'mM': list([])} 
        for c in range(num_classes):
            features['cprob'+str(c)] = list([])

        features['name'].append( item[2] )

        gt = item[1]

        if adv:
            probs = np.load(item[4])
        else:
            probs = np.load(item[3])
        seg = np.argmax(probs, axis=0)
        seg[gt==255] = 255

        counter_c = 0
        iou_all = 0
        for c in range(num_classes):
            I = np.sum(np.logical_and(seg==c,gt==c))
            U = np.sum(np.logical_or(seg==c,gt==c))
            if U > 0:
                counter_c += 1
                iou_all += I / U
        features['miou'].append( iou_all / counter_c )

        gt_pix = np.sum(gt!=255)
        features['acc'].append( np.sum(seg[seg!=255]==gt[gt!=255]) / gt_pix )
        features['apsr'].append( (gt_pix-np.sum(seg[seg!=255]==gt[gt!=255])) / gt_pix )

        features['mE'].append( np.mean(entropy(probs,axis=0)[gt!=255]) )
        features['mV'].append( np.mean(variation_ratio(probs)[gt!=255]) )
        features['mM'].append( np.mean(probdist(probs)[gt!=255]) )

        for c in range(num_classes):
            if np.sum(seg==c) > 0:
                features['cprob'+str(c)].append( np.sum( np.asarray(probs[c,seg==c],dtype='float64') ) / np.sum(seg==c) )
                assert(features['cprob'+str(c)][-1] != np.inf)
            else: 
                features['cprob'+str(c)].append( 0 )
 
        pickle.dump(features, open( save_path, 'wb' ) )


def concat_features():
    print('concat features')

    feature_path = CONFIG.COMP_FEATURES_DIR

    save_path_clean = feature_path + 'features_all_False.p'
    save_path_adv = feature_path + 'features_all_True.p'

    if os.path.isfile(save_path_clean) and os.path.isfile(save_path_adv):
        f_clean = pickle.load( open( save_path_clean, 'rb' ) )
        f_adv = pickle.load( open( save_path_adv, 'rb' ) )

    else:

        flag_first = True
        for file,i in zip(sorted(os.listdir(feature_path)),range(len(sorted(os.listdir(feature_path))))):
            if flag_first:
                f_adv = pickle.load( open( feature_path + file, 'rb' ) )
                f_clean = pickle.load( open( feature_path.replace(CONFIG.ATTACK+'/','') + file.replace('True','False'), 'rb' ) )
                flag_first = False
            else:
                f_adv_tmp = pickle.load( open( feature_path + file, 'rb' ) )
                f_clean_tmp = pickle.load( open( feature_path.replace(CONFIG.ATTACK+'/','') + file.replace('True','False'), 'rb' ) )
                for j in f_adv:
                    f_adv[j] += f_adv_tmp[j]
                    f_clean[j] += f_clean_tmp[j]
            
            sys.stdout.write('\t concatenated file number {} / {}\r'.format(i,len(os.listdir(feature_path))))

        pickle.dump(f_clean, open( save_path_clean, 'wb' ) )
        pickle.dump(f_adv, open( save_path_adv, 'wb' ) )

    return f_clean, f_adv


def plot_vio(df_full, key):
    f, ax = plt.subplots(figsize=(7,4))
    plt.clf() 
    ax = sns.violinplot(x='status', y=key, data=df_full, palette='Set3')
    plt.xlabel('')
    plt.ylabel(key)
    plt.tight_layout()
    plt.savefig(CONFIG.VIS_VIOS_DIR + 'vios_' + key + '.png', dpi=400, bbox_inches='tight')
    plt.close()


def plot_bars(f_clean, f_adv):

    mean_e = np.zeros((2,10))
    counter = 0
    for m in range(10,110,10):
        mean_e[0,counter] = np.mean(f_clean['E_'+str(m)])
        mean_e[1,counter] = np.mean(f_adv['E_'+str(m)])
        counter += 1
    
    X = np.arange(10)
    
    fig = plt.figure()
    plt.clf() 
    ax = fig.add_axes([0,0,1,1])
    ax.bar(X + 0.00, mean_e[0], color = 'lightseagreen', width = 0.25, label='clean')
    ax.bar(X + 0.25, mean_e[1], color = 'darkviolet', width = 0.25, label='attacked')
    ax.set_yscale('log')
    plt.xticks(X+0.125, ('0-10', '10-20', '20-30', '30-40', '40-50', '50-60', '60-70', '70-80', '80-90', '90-100'))
    plt.xlabel('entropy')
    plt.ylabel('averaged number of pixels')
    plt.legend()
    plt.savefig(CONFIG.VIS_VIOS_DIR + 'bars.png', dpi=400, bbox_inches='tight')
    plt.close()


def plot_features(f_clean, f_adv):

    status = ['clean'] * len(f_clean['mE']) + ['adversarial'] * len(f_adv['mE'])
    features_all = {'status': status}
    for key in f_clean.keys(): 
        features_all[key] = f_clean[key] + f_adv[key]

    pd.options.display.float_format = '{:,.5f}'.format
    df_full = pd.DataFrame( data=features_all )
    df_full = df_full.drop(['name'], axis=1)

    print('visualize vios')
    for key in df_full.drop(['status'], axis=1).keys(): 
        plot_vio(df_full, key)


def plot_histrogram(f_clean, f_adv, key, save_path=CONFIG.DETECT_HISTO_DIR):

    num_img = len(f_adv['name'])

    fig = plt.figure()
    plt.clf()
    if key in f_clean.keys():
        _ = plt.hist(f_clean[key], bins='auto', color='cornflowerblue', label='clean', alpha=.5)
        _ = plt.hist(f_adv[key], bins='auto', color='orchid', label='adv', alpha=.5)
    elif 'ocsvm' == key:
        scores = np.load(os.path.join(save_path, 'scores.npy'))
        _ = plt.hist(scores[:num_img], bins='auto', color='cornflowerblue', label='clean', alpha=.5)
        _ = plt.hist(scores[num_img:], bins='auto', color='orchid', label='adv', alpha=.5)
    plt.legend()
    fig.savefig(os.path.join(save_path, 'histogram_' + key + '.png'), dpi=400, bbox_inches='tight') #, transparent=True)
    plt.close()


def plot_detect_acc(f_clean, f_adv, key, num_steps=39, save_path=CONFIG.DETECT_HISTO_DIR):

    num_img = len(f_adv['name'])
    # clean&non-detect (TN), clean&detect (FP), adv&non-detect (FN), adv&detect (TP)
    detect = np.zeros((num_steps+1,4)) 

    if key in f_clean.keys():
 
        step_size = (np.max(f_clean[key])-np.min(f_clean[key]))/num_steps
        th_range = np.arange(np.min(f_clean[key]),np.max(f_clean[key]),step_size)
        if len(th_range) == num_steps:
            th_range = np.append(th_range,[th_range[-1]+step_size])
        assert len(th_range) == num_steps+1

        if np.mean(f_clean[key]) < np.mean(f_adv[key]):
            for th,t in zip(th_range,range(len(th_range))):
                for i in range(num_img):
                    if f_clean[key][i] < th:
                        detect[t,0] += 1
                    else:
                        detect[t,1] += 1
                    if f_adv[key][i] < th:
                        detect[t,2] += 1
                    else:
                        detect[t,3] += 1
        else:
            for th,t in zip(th_range,range(len(th_range))):
                for i in range(num_img):
                    if f_clean[key][i] > th:
                        detect[t,0] += 1
                    else:
                        detect[t,1] += 1
                    if f_adv[key][i] > th:
                        detect[t,2] += 1
                    else:
                        detect[t,3] += 1

    elif key in ['ocsvm', 'ee', 'mV']:
        scores = np.load(os.path.join(save_path, 'scores_'+key+'.npy'))
        scores = (scores - np.min(scores)) / (np.max(scores) - np.min(scores))
        if np.mean(scores[:num_img]) > np.mean(scores[num_img:]):
            scores = 1-scores

        step_size = 1/num_steps
        th_range = np.sort(scores)[0::int(2*num_img/(num_steps-1))]
        th_range = np.append(th_range,np.sort(scores)[-1])

        for th,t in zip(th_range,range(len(th_range))):
            for i in range(num_img): 
                if scores[i] < th:
                    detect[t,0] += 1
                else:
                    detect[t,1] += 1
                if scores[i+num_img] < th:
                    detect[t,2] += 1  
                else:
                    detect[t,3] += 1
    
    else:
        ya_pred = np.load(os.path.join(save_path, 'ya_pred_' + key + '.npy'))[:,1]

        if np.mean(ya_pred[:num_img]) > np.mean(ya_pred[num_img:]):
            ya_pred = 1-ya_pred

        step_size = 1/num_steps
        th_range = np.sort(ya_pred)[0::int(2*num_img/(num_steps-1))]
        th_range = np.append(th_range,np.sort(ya_pred)[-1])

        for th,t in zip(th_range,range(len(th_range))):
            for i in range(num_img): 
                if ya_pred[i] < th:
                    detect[t,0] += 1
                else:
                    detect[t,1] += 1
                if ya_pred[i+num_img] < th:
                    detect[t,2] += 1  
                else:
                    detect[t,3] += 1
        
    tpr = detect[:,3] / (detect[:,3] + detect[:,2])
    fpr = detect[:,1] / (detect[:,0] + detect[:,1])
    fpr5 = np.zeros((len(fpr))) + 0.05
    idx_th_tpr = np.argmin(np.abs(fpr-fpr5))
    F1  = (2*detect[:,3]) / (2*detect[:,3] + detect[:,1] + detect[:,2])

    detect_rate = detect / num_img
                
    result_path = os.path.join( save_path, 'results_detect_th_' + key + '.txt')
    with open(result_path, 'wt') as fi:
        print('feature: ', key, file=fi)
        for th,t in zip(th_range,range(len(th_range))):
            print('threshold: ', th, file=fi)
            print('clean & non-detect', detect[t,0], detect_rate[t,0], file=fi)
            print('clean & detect', detect[t,1], detect_rate[t,1], file=fi)
            print('adv & non-detect', detect[t,2], detect_rate[t,2], file=fi)
            print('adv & detect', detect[t,3], detect_rate[t,3], file=fi)
            print(' ', file=fi)
        
    result_path = os.path.join( save_path, 'results_detect_max_' + key + '.txt')
    with open(result_path, 'wt') as fi:
        print('feature: ', key, file=fi)
        mi = np.argmax(detect_rate[:,0] + detect_rate[:,3])
        print('clean & non-detect', detect[mi,0], detect_rate[mi,0], file=fi)
        print('clean & detect', detect[mi,1], detect_rate[mi,1], file=fi)
        print('adv & non-detect', detect[mi,2], detect_rate[mi,2], file=fi)
        print('adv & detect', detect[mi,3], detect_rate[mi,3], file=fi)
        print('max detection accuracy', detect_rate[mi,0] + detect_rate[mi,3], file=fi)
        print('max averaged detection accuracy', (detect_rate[mi,0]+detect_rate[mi,3])/2, file=fi)
        print('AUC', auc(fpr, tpr), file=fi)
        min_else = 0
        if len(tpr[tpr!=0]) > 0:
            min_else = np.min(tpr[tpr!=0])
        print('TPR_5%', tpr[idx_th_tpr], 'th', th_range[idx_th_tpr], 'min else:', min_else, file=fi)
        print('mean F1', np.mean(F1), file=fi)
        print('max F1', np.max(F1), file=fi)
        print(' ', file=fi)
    
    fig = plt.figure()
    plt.clf()
    ax = fig.add_axes([0,0,1,1])
    ax.bar(th_range, detect_rate[:,0], 0.8*step_size, color='cornflowerblue', label='clean & non-detected')
    ax.bar(th_range, detect_rate[:,3], 0.8*step_size, bottom=detect_rate[:,0], color='mediumvioletred', label='adv & detected')
    ax.set_xlabel('thresholds')
    ax.set_ylabel('detection accuracy')
    ax.legend()
    fig.savefig(os.path.join(save_path, 'detect_acc_' + key + '.png'), dpi=400, bbox_inches='tight') #, transparent=True)
    plt.close()

    fig = plt.figure()
    plt.clf()
    plt.plot(th_range, (detect_rate[:,0]+detect_rate[:,3])/2, marker='o', color='palevioletred')
    plt.xlabel('thresholds')
    plt.ylabel('averaged detection accuracy')
    fig.savefig(os.path.join(save_path, 'aver_detect_acc_' + key + '.png'), dpi=400, bbox_inches='tight') #, transparent=True)
    plt.close()

    fig = plt.figure()
    plt.clf()
    plt.plot(fpr, tpr, marker='o', color='lightseagreen')
    plt.xlabel('false positive rate')
    plt.ylabel('true positive rate')
    plt.title('AUC:'+str(auc(fpr, tpr))+' - TPR_5%:'+str(tpr[idx_th_tpr]))
    fig.savefig(os.path.join(save_path, 'auc_' + key + '.png'), dpi=400, bbox_inches='tight') #, transparent=True)
    plt.close()

    fig = plt.figure()
    plt.clf()
    plt.plot(th_range, F1, marker='o', color='hotpink')
    plt.xlabel('thresholds')
    plt.ylabel('F1')
    fig.savefig(os.path.join(save_path, 'F1_' + key + '.png'), dpi=400, bbox_inches='tight') #, transparent=True)
    plt.close()


def detect_adv(f_clean, f_adv):

    result_path = os.path.join( CONFIG.DETECT_HISTO_DIR, 'num_img.txt')
    with open(result_path, 'wt') as fi:
        print('number of images: ', len(f_clean['name']), file=fi)
        print(' ', file=fi)

    for key in f_clean.keys(): 
        if key in ['name', 'miou', 'acc', 'apsr'] or 'cprob' in key:
            continue
        plot_histrogram(f_clean, f_adv, key)
        plot_detect_acc(f_clean, f_adv, key)


def metrics_to_nparray(metrics, names):

    I = range(len(metrics['mE']))
    M = np.asarray( [ np.asarray(metrics[ m ])[I] for m in names ] )
    M = np.squeeze(M.T)
    return M


def metrics_to_dataset(f_clean, f_adv):

    X_names = sorted([ m for m in f_clean.keys() if m not in ['name','miou','acc','apsr']])

    Xa_clean = metrics_to_nparray(f_clean, X_names)
    Xa_adv = metrics_to_nparray(f_adv, X_names)
    assert np.any(np.isnan(Xa_clean)) == False and np.any(np.isinf(Xa_clean)) == False
    assert np.any(np.isnan(Xa_adv)) == False and np.any(np.isinf(Xa_adv)) == False
    Xa = np.concatenate((Xa_clean,Xa_adv),axis=0)
    ya = np.concatenate((np.zeros((Xa_clean.shape[0])),np.ones((Xa_adv.shape[0]))),axis=0)

    return Xa, ya, X_names


def train_outlier(X_train):
    # Unsupervised Outlier Detection
    model_ocsvm = OneClassSVM(nu=0.5,kernel='linear')
    model_ocsvm.fit( X_train ) # 1 inlier, -1 outlier

    # EllipticEnvelope assumes that the data should be Gaussian distributed
    model_ee = EllipticEnvelope(random_state=1,contamination=0.5)
    model_ee.fit( X_train )

    return model_ocsvm, model_ee


def init_stats(runs, X_names):
        '''
        initialize dataframe for storing results
        '''
        stats = dict({})
        single_stats = ['val_acc','val_auroc','train_acc','train_auroc']

        for s in single_stats:
            stats[s] = np.zeros((runs,))
        
        stats['coefs_classif'] = np.zeros((runs,len(X_names) ))
        stats['n_av']         = runs
        stats['n_metrics']    = len(X_names) 
        stats['metric_names'] = X_names
        
        return stats

def classification_fit_and_predict(X_train, y_train, X_val, y_val):

    print('classification input:', np.shape(X_train), np.shape(y_train), np.shape(X_val), np.shape(y_val))
    model = linear_model.LogisticRegression(penalty='none', solver='lbfgs', max_iter=5000, tol=1e-3)

    model.fit( X_train, y_train )

    y_train_pred = model.predict_proba(X_train)
    fpr, tpr, _ = roc_curve(y_train.astype(int),y_train_pred[:,1])
    print('model train auroc score:', auc(fpr, tpr) )
    y_val_pred = model.predict_proba(X_val)
    fpr, tpr, _ = roc_curve(y_val.astype(int),y_val_pred[:,1])
    print('model test auroc score:', auc(fpr, tpr) )
    return y_val_pred, y_train_pred, model


def fit_model_run(Xa_train, ya_train, Xa_val, ya_val, stats, run=0):

    ya_val_pred, ya_train_pred, model_classif = classification_fit_and_predict( Xa_train, ya_train, Xa_val, ya_val )

    stats['train_acc'][run] = np.mean( np.argmax(ya_train_pred,axis=-1)==ya_train )
    stats['val_acc'][run] = np.mean( np.argmax(ya_val_pred,axis=-1)==ya_val )

    fpr, tpr, _ = roc_curve(ya_train, ya_train_pred[:,1])
    stats['train_auroc'][run] = auc(fpr, tpr)
    fpr, tpr, _ = roc_curve(ya_val, ya_val_pred[:,1])
    stats['val_auroc'][run] = auc(fpr, tpr)

    stats['coefs_classif'][run] = np.array(model_classif.coef_)

    return stats, ya_val_pred, model_classif


def stats_dump(stats, ya_pred, attack_train, save_path=CONFIG.DETECT_CLASSIF_DIR):

    pickle.dump( stats, open( os.path.join(save_path, 'stats_' + attack_train + '.p'), "wb" ) )
    np.save(os.path.join(save_path, 'ya_pred_' + attack_train + '.npy'), ya_pred)

    with open(os.path.join(save_path, 'av_results_' + attack_train + '.txt'), 'wt') as f:
        print(attack_train, file=f)

        mean_stats = dict({})
        std_stats = dict({})

        for s in stats:
            if s not in ["n_av", "n_metrics", "metric_names"]:
                mean_stats[s] = np.mean(stats[s], axis=0)
                std_stats[s]  = np.std( stats[s], axis=0)
            
        print( "           & train                &  val  \\\\ ", file= f)
        M = sorted([ s for s in mean_stats if 'acc' in s ])
        print( "ACC       ", end=" & ", file= f )
        for s in M: print( "${:.2f}\%".format(100*mean_stats[s])+"(\pm{:.2f}\%)$".format(100*std_stats[s]), end=" & ", file=f )
        print("   \\\\ ", file=f )
            
        M = sorted([ s for s in mean_stats if 'auroc' in s ])
        print( "AUROC     ", end=" & ", file= f )
        for s in M: print( "${:.2f}\%".format(100*mean_stats[s])+"(\pm{:.2f}\%)$".format(100*std_stats[s]), end=" & ", file=f )
        print("   \\\\ ", file=f )
        print(" ", file=f )


def determine_counter_attacks():

    counter_attacks = list([])
    for i in [2,4,8,16]:
        counter_attacks.append('FGSM_untargeted'+str(i))
    for i in [2,4,8,16]:
        counter_attacks.append('FGSM_targeted'+str(i))
    for i in [2,4,8,16]:
        counter_attacks.append('FGSM_untargeted_iterative'+str(i))
    for i in [2,4,8,16]:
        counter_attacks.append('FGSM_targeted_iterative'+str(i))
    counter_attacks.append('smm_static')
    if CONFIG.DATASET == 'cityscape':
        counter_attacks.append('smm_dynamic')
        counter_attacks.append('patch_eot')

    counter_attacks.remove(CONFIG.ATTACK)
    print(counter_attacks)

    if CONFIG.MODEL_NAME == 'DeepLabV3_Plus_xception65':
        network_patch = 'ddrnet23Slim'
    elif CONFIG.MODEL_NAME == 'ddrnet23Slim':
        network_patch = 'DeepLabV3_Plus_xception65'
    elif CONFIG.MODEL_NAME == 'HRNet_hrnet_w18_small_v1':
        network_patch = 'bisenetX39'
    elif CONFIG.MODEL_NAME == 'bisenetX39':
        network_patch = 'HRNet_hrnet_w18_small_v1'

    return counter_attacks, network_patch




