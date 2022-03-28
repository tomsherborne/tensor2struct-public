import math
import torch
import torch.nn as nn
import collections

import copy
import logging

logger = logging.getLogger("tensor2struct")


class FirstOrderModelAgnosticMetaLearning(nn.Module):
    def __init__(
        self, model=None, inner_opt=None, first_order=False, device=None,
    ):
        super().__init__()
        self.inner_opt = inner_opt
        self.first_order = first_order
        self.inner_steps = 1
        self.device = device

    def get_inner_opt_params(self):
        """
        Equvalent to self.parameters()
        """
        return []

    def meta_train(self, model, inner_batch, outer_batches):
        if self.first_order:
            return self.fmaml_train(model, inner_batch, outer_batches)
        else:
            return self.maml_train(model, inner_batch, outer_batches)

    def fmaml_train(self, model, inner_batch, outer_batches):
        logger.info("Using FMAML ->>")

        assert len(outer_batches) == 1  # support list later

        # state_dict won't give u the gradients
        theta_0 = copy.deepcopy(list(model.get_trainable_parameters()))
        assert len(self.inner_opt.param_groups) == 1
        lr = self.inner_opt.param_groups[0]["lr"]
        assert lr > 1e-8
        # logger.info(f"inner lr: {lr}")

        # meta train
        self.inner_opt.zero_grad()
        inner_loss = model(inner_batch)["loss"]
        inner_loss.backward()
        self.inner_opt.step()
        logger.info(f"Inner loss: {inner_loss.item()}")

        # meta-test
        self.inner_opt.zero_grad()
        outer_loss = model(outer_batches[0])["loss"]
        outer_loss.backward()
        logger.info(f"Outer loss: {outer_loss.item()}")

        # update
        theta = list(model.get_trainable_parameters())
        for param_0, param in zip(theta_0, theta):
            # get back to theta_0 (note that this is WRONG but we keep it here as Bailin's version)

            param.data.copy_(param_0.data)

            if not param.requires_grad:
                continue
            with torch.no_grad():
                # sum gradients
                # NOTE that I think the LR adjustment here is bullshit. 
                src_grad = (param_0 - param) / (lr + 1e-8) # undoing existing LR adjustment here. 
                
                tgt_grad = param.grad
                if tgt_grad is not None:
                    param.grad = src_grad + tgt_grad
                else:
                    # logger.warn("No target-domain gradient found")
                    param.grad = src_grad

        del theta_0
        assert model.training
        return {"loss": (inner_loss.item() + outer_loss.item()) / 2}

    def maml_train(self, model, inner_batch, outer_batches):
        logger.info("Using MAML ->>")
        assert model.training
        ret_dic = {}
        with higher.innerloop_ctx(
            model, self.inner_opt, copy_initial_weights=False, device=self.device
        ) as (fmodel, diffopt), torch.backends.cudnn.flags(enabled=False):
            for _step in range(self.inner_steps):
                inner_ret_dic = fmodel(inner_batch)
                inner_loss = inner_ret_dic["loss"]

                # use the snippet for checking higher
                # def test(params):
                #     params = [p for p in params if p.requires_grad]
                #     all_grads = torch.autograd.grad(
                #         loss,
                #         params,
                #         retain_graph=True,
                #         allow_unused=True,
                #     )
                #     print(len(params), sum(p is not None for p in all_grads))
                # import pdb; pdb.set_trace()
                # test(model.parameters())
                # test(fmodel.fast_params)

                diffopt.step(inner_loss)
            logger.info(f"Inner loss: {inner_loss.item()}")

            mean_outer_loss = torch.Tensor([0.0]).to(self.device)
            with torch.set_grad_enabled(model.training):
                for batch_id, outer_batch in enumerate(outer_batches):
                    outer_ret_dic = fmodel(outer_batch)
                    mean_outer_loss += outer_ret_dic["loss"]
            mean_outer_loss.div_(len(outer_batches))
            logger.info(f"Outer loss: {mean_outer_loss.item()}")

            final_loss = inner_loss + mean_outer_loss
            final_loss.backward()

            # not sure if it helps
            del fmodel
            import gc

            gc.collect()

        ret_dic["loss"] = final_loss.item()
        return ret_dic

    # @torch.no_grad()
    # def sgd_step(self):
    #     """
    #     Copy from torch.optim.sgd, ATTENTION: it assume that the order of 
    #     masks is the same as  the order of paramters, do check it!
    #     """

    #     loss = None

    #     assert len(self.inner_opt.param_groups) == 1
    #     group = self.inner_opt.param_groups[0]

    #     weight_decay = group["weight_decay"]
    #     momentum = group["momentum"]
    #     dampening = group["dampening"]
    #     nesterov = group["nesterov"]

    #     for i, p in enumerate(group["params"]):
    #         if p.grad is None:
    #             continue

    #         d_p = p.grad

    #         if weight_decay != 0:
    #             d_p = d_p.add(p, alpha=weight_decay)
    #         if momentum != 0:
    #             param_state = self.state[p]
    #             if "momentum_buffer" not in param_state:
    #                 buf = param_state["momentum_buffer"] = torch.clone(d_p).detach()
    #             else:
    #                 buf = param_state["momentum_buffer"]
    #                 buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
    #             if nesterov:
    #                 d_p = d_p.add(buf, alpha=momentum)
    #             else:
    #                 d_p = buf

    #         p.add_(d_p, alpha=-group["lr"])

    #     return loss

    # def meta_test(self, model, inner_batch):
    #     assert model.training

    #     self.inner_opt.zero_grad()
    #     ret_dict = model(inner_batch)
    #     loss = ret_dict["loss"]
    #     loss.backward()

    #     # gradient step
    #     if isinstance(self.inner_opt, torch.optim.Adam):
    #         self.adam_step()
    #     elif isinstance(self.inner_opt, torch.optim.SGD):
    #         self.sgd_step()
    #     else:
    #         raise NotImplementedError

    #     logger.info(f"Meta-test loss: {loss.item()}")

    #     return ret_dict


MAML = FirstOrderModelAgnosticMetaLearning
