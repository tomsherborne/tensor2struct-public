from argparse import ArgumentError
import math
from typing import Dict
from numpy import inner, mean, prod
import torch
import torch.nn as nn
import collections
import gc

import copy
import logging

logger = logging.getLogger("tensor2struct")


class DomainGeneralizationReptileMetaLearning(nn.Module):
    def __init__(
        self, 
        model=None, 
        inner_opt=None, 
        inner_steps=1, 
        first_order=True, 
        do_cross_gradient_normalization=False,
        device=None,
    ):
        super().__init__()
        self.inner_opt = inner_opt
        self.first_order = first_order
        self.inner_steps = inner_steps
        self.do_cross_gradient_normalization = do_cross_gradient_normalization # PCGrad+FROG Gradient Scaling
        self.device = device
        self.first_order_train = self.dgreptile_train if do_cross_gradient_normalization else self.dgreptile_train_without_norm
        logger.info(f"Running Reptile with K={self.inner_steps} steps")

    def get_inner_opt_params(self):
        """
        Equvalent to self.parameters()
        """
        return []

    def meta_train(self, model, inner_batches, outer_batch):
        if self.first_order:
            return self.first_order_train(model, inner_batches, outer_batch)
        else:
            raise ArgumentError("MAML training is not supported for {self.__class__}")


    def dgreptile_train(self, model, inner_batches, outer_batch):
        """
        Runs K inner batches and 1 outer batch. Applies a domain-generalisation cross-database 
        step between Reptile-approx of inner-manifold and outer-step.

        If do_cross_gradient_normalization is true. The cross-gen step is normalized
        """
        logger.info("Using DGREPTILE with gradient normalization ->>")

        # state_dict won't give gradients
        theta_0 = copy.deepcopy(list(model.get_trainable_parameters()))

        # Inner step only updates the decoder params
        assert len(self.inner_opt.param_groups) == 1
        lr = self.inner_opt.param_groups[0]["lr"]
        assert lr > 1e-8

        # meta train
        mean_inner_loss = torch.Tensor([0.0]).to(self.device)
        # Each iter is a single step
        for batch in inner_batches:
            self.inner_opt.zero_grad()
            batch_loss = model(batch)["loss"]
            batch_loss.backward()
            self.inner_opt.step()
            mean_inner_loss += batch_loss.detach()
            logger.info(f"Inner loss: {batch_loss.item()}")

        mean_inner_loss.div_(len(inner_batches)) # Mean inner loss
        logger.info(f"Mean Inner loss: {mean_inner_loss.item()}")
        self.inner_opt.zero_grad()

        # Outer batch in other domain
        outer_loss = model(outer_batch[0])["loss"]
        outer_loss.backward() # populate .grad with outer step gradients
        logger.info(f"Outer loss: {outer_loss.item()}")
        
        theta = list(model.get_trainable_parameters())

        # ###################################################################
        # # PC-GRAD - Adapted from https://github.com/OrthoDex/PCGrad-PyTorch/blob/master/pcgrad.py
        # ###################################################################

        src_grads = []
        tgt_grads = []
        grd_shape = []

        # Collect gradients (src and tgt)
        with torch.no_grad():
            for param_0, param in zip(theta_0, theta):
                src_grad = param - param_0
                src_grads.append(src_grad)

                tgt_grad = param.grad
                if type(tgt_grad) == type(None): # 
                    tgt_grad = torch.zeros_like(src_grad, device=src_grad.device)
                tgt_grads.append(tgt_grad)
                grd_shape.append(tuple(tgt_grad.shape))

            src_grads_flat = torch.cat([g.reshape(-1) for g in src_grads])
            tgt_grads_flat = torch.cat([g.reshape(-1) for g in tgt_grads])

            cos_product = torch.dot(src_grads_flat, tgt_grads_flat)
            logger.info(f"Cosine {cos_product}")
            if cos_product < 0:
                logger.info(f"Cosine conflict detected {cos_product}. Running PCGrad gradient alignment...")
                del src_grads, tgt_grads
                torch.cuda.empty_cache()

                # D_s = cos_product / tgt.tgt
                D_s = cos_product / tgt_grads_flat.norm()**2
                # D_t = cos_product / src.src
                D_t = cos_product / src_grads_flat.norm()**2

                src_grads_tmp = src_grads_flat.clone()
                src_grads_flat.sub_(tgt_grads_flat * D_s)
                tgt_grads_flat.sub_(src_grads_tmp * D_t)
                del src_grads_tmp, D_s, D_t   # Pessimistic cleanup
                torch.cuda.empty_cache()

                # We want to do a concat+sum over src+tgt per param
                src_grads_flat.add_(tgt_grads_flat)

                #   Unpack Grads back into model and pack together...
                combined_grad, idx = [], 0
                for shape in grd_shape:
                    ln = prod(shape)
                    combined_grad.append(src_grads_flat[idx: idx + ln].view(shape))
                    idx += ln

                # Overly pessimistic gradient cleanup
                del src_grads_flat, tgt_grads_flat, cos_product
                torch.cuda.empty_cache()
            else:
                # We don't need PCGrad calculations if cos product is < 0
                # In this case the combined gradient is the sum of src and tgt without adjustment
                combined_grad = [torch.add(s, t) for s, t in zip(src_grads, tgt_grads)]
                
                del src_grads, tgt_grads, src_grads_flat, tgt_grads_flat
            # src_norm = torch.norm(src_grads_flat)
            # tgt_norm = torch.norm(tgt_grads_flat)
            # sum_norm = src_norm + tgt_norm
            # tgt_scaler = src_norm / sum_norm
            # src_scaler = tgt_norm / sum_norm

            # combined_grad = [torch.add(s * src_scaler, t * tgt_scaler)
            #                     for s, t in zip(src_grads, tgt_grads)]
            
            torch.cuda.empty_cache()

            # update
            for param_0, param, adjusted_grad in zip(theta_0, theta, combined_grad):
                
                if not param.requires_grad:
                    logger.warning(f"Skipping param outer-loop adjustment: {param}")
                    continue

                # Reset data arg back to theta_0
                param.data.copy_(param_0.data)
                
                # Set to PC-Grad adjusted gradient or Normed
                param.grad = adjusted_grad

        # Cleanup
        del theta_0
        gc.collect()
        assert model.training
        
        # This is for reporting only not .step()
        return {"loss": (mean_inner_loss.item() + outer_loss.item()) / 2} 

    # COPY OF OLD VERSION OF DGREPTILE TRAINING WHICH WE KNOW WORKS BEFORE I MESS EVERYTHING UP WITH PCGRAD
    def dgreptile_train_without_norm(self, model, inner_batches, outer_batch):
        logger.info("Using DGREPTILE ->>")

        # state_dict won't give u the gradients
        theta_0 = copy.deepcopy(list(model.get_trainable_parameters()))

        # Inner step only updates the decoder params
        assert len(self.inner_opt.param_groups) == 1
        lr = self.inner_opt.param_groups[0]["lr"]
        assert lr > 1e-8
        
        # meta train
        mean_inner_loss = torch.Tensor([0.0]).to(self.device)
        # Each iter is a single step
        for batch in inner_batches:
            self.inner_opt.zero_grad()
            batch_loss = model(batch)["loss"]
            batch_loss.backward()
            self.inner_opt.step()
            mean_inner_loss += batch_loss.detach()
            logger.info(f"Inner loss: {batch_loss.item()}")

        mean_inner_loss.div_(len(inner_batches)) # Mean inner loss
        logger.info(f"Mean Inner loss: {mean_inner_loss.item()}")
        self.inner_opt.zero_grad()
        
        # We know the reptile only version was a bust so outer_batch should always be present
        if outer_batch:
            # meta-test
            outer_loss = model(outer_batch[0])["loss"]
            outer_loss.backward() # populate .grad with outer step gradients
            logger.info(f"Outer loss: {outer_loss.item()}")
        else:
            logger.warning("No Outer Batch detected. This shouldn't happen so something is wrong")
            outer_loss = mean_inner_loss # dummy so the return statement is neat
            
        # update
        theta = list(model.get_trainable_parameters())
        for param_0, param in zip(theta_0, theta):
            if param.requires_grad:    
                with torch.no_grad():
                    # sum gradients
                    src_grad = param - param_0 # Reptile difference step
                    tgt_grad = param.grad

                    if tgt_grad is not None:
                        param.grad = src_grad + tgt_grad
                    else:
                        param.grad = src_grad

            # Reset data arg back to theta_0
            param.data.copy_(param_0.data)

        # Cleanup
        del theta_0
        gc.collect()
        assert model.training
        
        # This is for reporting only not .step()
        return {"loss": (mean_inner_loss.item() + outer_loss.item()) / 2} 

    def maml_train(self, model, inner_batch, outer_batches) -> Dict:
        raise ArgumentError("MAML Training is not supported using DG-Reptile")
        return {}


MAML = DomainGeneralizationReptileMetaLearning
