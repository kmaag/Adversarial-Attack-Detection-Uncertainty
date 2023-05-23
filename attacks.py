import torch
import numpy as np
import torch.nn as nn
from tqdm import trange
from torch.autograd import Variable
from scipy.interpolate import NearestNDInterpolator


class Attacks():
    """
    Attacks class contains different attacks 
    """
    def __init__(self, model, device='cpu'): 
        """ Initialize the attack class
        Args:
            model (torch.nn model): model to be attacked
            device  (device):       device
        """
        self.model = model
        self.device = device
    

    def model_pred(self, img):
        """ model prediction
        Args:
            img (torch.tensor): input image
        Returns:
           pred (torch.tensor): predicted semantic segmentation
        """
        pred = self.model.evaluate(img)
        return pred


    def FGSM_untargeted(self, img, label, eps=4):
        """ FGSM untargeted attack (FGSM)
        Args:
            img    (torch.tensor): input image
            label  (torch.tensor): label of the input image
            eps  (float):          size of adversarial perturbation
        Returns:
           adv_img (torch.tensor): pertubed image
           noise   (torch.tensor): adversarial noise
        """

        eps = eps / 255
        loss = nn.CrossEntropyLoss(ignore_index=-1)

        self.model.zero_grad()
        pred = self.model_pred(img)
            
        lo = loss(pred, label.detach())
        lo.backward()
        im_grad = img.grad
        assert(im_grad is not None)

        noise = eps * torch.sign(im_grad)
        adv_img = img + noise
        return adv_img, noise 


    def FGSM_targeted(self, img, eps=4):
        """ FGSM targeted attack (FGSM ll)
        Args:
            img    (torch.tensor): input image
            eps  (float):          size of adversarial perturbation
        Returns:
           adv_img (torch.tensor): pertubed image
           noise   (torch.tensor): adversarial noise
        """

        eps = eps / 255
        loss = nn.CrossEntropyLoss(ignore_index=-1)

        self.model.zero_grad()
        pred = self.model_pred(img)
        target = (torch.argmin(pred[0],0)).unsqueeze(0)

        lo = loss(pred, target.detach()) 
        lo.backward()
        im_grad = img.grad
        assert(im_grad is not None)

        noise = -eps * torch.sign(im_grad)
        adv_img = img + noise
        return adv_img, noise


    def FGSM_untargeted_iterative(self, img, label, alpha=1, eps=4, num_it=None):
        """ FGSM iterative untargeted (I-FGSM)
        Args:
            img    (torch.tensor): input image
            label  (torch.tensor): label 
            alpha  (float):        step size of the attack
            eps    (float):        size of adversarial perturbation
            num_it (int):          number of attack iterations 
        Returns:
           adv_img (torch.tensor): pertubed image
           noise   (torch.tensor): adversarial noise
        """

        if num_it == None:
            num_it = min(int(eps+4), int(1.25*eps))
        eps = eps / 255
        alpha = alpha / 255
        loss = nn.CrossEntropyLoss(ignore_index=-1)

        adv_img = img 
        adv_img.requires_grad = True
        
        tbar=trange(num_it)
        for i in tbar:

            self.model.zero_grad()

            pred = self.model_pred(adv_img)

            lo = loss(pred, label.detach())
            lo.backward()
            im_grad = adv_img.grad
            assert(im_grad is not None)

            noise = (alpha * torch.sign(im_grad)).clamp(-eps,eps)
            adv_img = (adv_img + noise).clamp(img-eps,img+eps)
            adv_img = Variable(adv_img, requires_grad=True)

            tbar.set_description('Iteration: {}/{} of I-FGSM attack'.format(i, num_it))
        return adv_img, noise 


    def FGSM_targeted_iterative(self, img, alpha=1, eps=4, num_it=None):
        """ FGSM iterative least likely class (targeted) attack (I-FGSM ll)
        Args:
            img    (torch.tensor): input image
            alpha  (float):        step size of the attack
            eps    (float):        size of adversarial perturbation
            num_it (int):          number of attack iterations 
        Returns:
           adv_img (torch.tensor): pertubed image
           noise   (torch.tensor): adversarial noise
        """

        if num_it == None:
            num_it = min(int(eps+4), int(1.25*eps))
        eps = eps / 255
        alpha = alpha / 255
        loss = nn.CrossEntropyLoss(ignore_index=-1)

        with torch.no_grad():
            pred = self.model_pred(img)
        target = (torch.argmin(pred[0],0)).unsqueeze(0)

        adv_img = img 
        adv_img.requires_grad = True

        tbar = trange(num_it)
        for i in tbar:

            self.model.zero_grad()

            pred = self.model_pred(adv_img)

            lo = loss(pred, target.detach()) 
            lo.backward()
            im_grad = adv_img.grad
            assert(im_grad is not None)

            noise = (-alpha * torch.sign(im_grad)).clamp(-eps,eps)
            adv_img = (adv_img + noise).clamp(img-eps,img+eps)
            adv_img = Variable(adv_img, requires_grad=True)

            tbar.set_description('Iteration: {}/{} of I-FGSM ll attack'.format(i, num_it))
        return adv_img, noise 
    

    def universal_adv_pert_static(self, train_loader, target_img_name=None, alpha=0.01, eps=0.1, num_it=60):
        """ Computes the universal adversarial pertubations (static)
        Args:
            train_loader    (dataloader):   training data
            target_img_name (path):         target image filename
            alpha           (float):        step size of the attack
            eps             (float):        size of adversarial perturbation
            num_it          (int):          number of attack iterations 
        Returns:
           noise            (torch.tensor): adversarial noise
        """

        loss = nn.CrossEntropyLoss(ignore_index=-1,reduction='none')

        # create static target
        for i, (image, label, filename) in enumerate(train_loader):
            if i == 0:
                _, _, H, W = image.size()
                if target_img_name == None:
                    target_img = image.clone()
                    break
            if target_img_name == filename[0]:
                target_img = image.clone()
                break
        print('Target image filename:', filename)
        target_img = target_img.to(self.device)
        with torch.no_grad():
            pred = self.model_pred(target_img)
        target = (torch.argmax(pred[0],0)).unsqueeze(0)
            
        num_img = len(train_loader)
        noise = torch.zeros(image.size())
        noise = noise.to(self.device)

        for it in trange(num_it):

            sum_grads = torch.zeros(noise.size()).to(self.device)

            for i, (image, label, filename) in enumerate(train_loader):

                self.model.zero_grad()

                image = image.to(self.device)
                image.requires_grad = True

                pred = self.model_pred(image + noise)

                J_cls = loss(pred, target.detach()) 

                # loss of target pixels predicted as desired class with confidence above tau is set to 0
                pred_softmax = torch.softmax(pred[0],0)
                J_cls[0,torch.logical_and(torch.argmax(pred_softmax,0)==target.squeeze(0), torch.max(pred_softmax,0)[0]>0.75)] = 0

                J_ss = torch.sum(J_cls) / (H * W)

                J_ss.backward()
                im_grad = image.grad
                assert(im_grad is not None)

                sum_grads = sum_grads + im_grad

            noise = (noise - alpha * torch.sign(sum_grads / num_img)).clamp(-eps,eps)
        
        print("Total training samples: {:d}".format(num_img))
        
        return noise


    def universal_adv_pert_dynamic(self, train_loader, class_id=11, alpha=0.01, eps=0.1, num_it=60):
        """ Computes the universal adversarial pertubations (dynamic)
        Args:
            train_loader (dataloader):   training data
            class_id     (int):          removing class id
            alpha        (float):        step size of the attack
            eps          (float):        clipping value for iterative attacks
            num_it       (int):          number of attack iterations 
        Returns:
           noise         (torch.tensor): adversarial noise
        """

        loss = nn.CrossEntropyLoss(ignore_index=-1,reduction='none') 

        for i, (image, label, filename) in enumerate(train_loader):
            _, _, H, W = image.size()
            break

        num_img = 0
        noise = torch.zeros(image.size())
        noise = noise.to(self.device)
        preds_all = torch.zeros((len(train_loader), H, W), dtype=torch.uint8).to(self.device)
        targets_all = torch.zeros((len(train_loader), H, W), dtype=torch.long).to(self.device)

        for it in trange(num_it):

            sum_grads = torch.zeros(noise.size()).to(self.device)

            for i, (image, label, filename) in enumerate(train_loader):

                # use image for training if class_id is included
                if class_id in torch.unique(label):
                    
                    self.model.zero_grad()

                    image = image.to(self.device)
                    image.requires_grad = True

                    # create dynamic target for each image ones
                    if it == 0:
                        num_img = num_img + 1

                        with torch.no_grad():
                            pred1 = self.model_pred(image)
                        pred_orig = (torch.argmax(pred1[0],0))
                        preds_all[i] = pred_orig.clone()
                        pred_orig_np = pred_orig.cpu().data.numpy()
                        pred_orig_np[pred_orig_np==class_id] = -2
                        mask = np.where(~(pred_orig_np==-2))
                        interp = NearestNDInterpolator(np.transpose(mask), pred_orig_np[mask])
                        pred_filled_np = interp(*np.indices(pred_orig_np.shape))
                        targets_all[i] = torch.from_numpy(pred_filled_np) 

                    pred = self.model_pred(image + noise)

                    J_cls = loss(pred, targets_all[i].unsqueeze(0).detach()) 

                    # loss of target pixels predicted as deesired class with confidence above tau is set to 0
                    pred_softmax = torch.softmax(pred[0],0)
                    J_cls[0,torch.logical_and(torch.argmax(pred_softmax,0)==targets_all[i], torch.max(pred_softmax,0)[0]>0.75)] = 0

                    w = 0.9999 
                    J_ss_w = (w * torch.sum(J_cls[0,preds_all[i]==class_id]) + (1-w) * torch.sum(J_cls[0,preds_all[i]!=class_id])) / (H * W)

                    J_ss_w.backward()
                    im_grad = image.grad
                    assert(im_grad is not None)

                    sum_grads = sum_grads + im_grad
                
            noise = (noise - alpha * torch.sign(sum_grads / num_img)).clamp(-eps,eps)

        print("Total training samples: {:d}".format(num_img))

        return noise


    def segmentation_mask_method(self, img, noise):
        """ Static or dynamic segmentation mask method
        Args:
            img    (torch.tensor): input image
            noise  (torch.tensor): universal adversarial noise
        Returns:
           adv_img (torch.tensor): pertubed image
        """

        adv_img = img + noise
        return adv_img

        