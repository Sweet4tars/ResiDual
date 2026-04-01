import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import lib.utils as utils

from lib.encoders import get_image_encoder, get_text_encoder
from lib.loss import loss_select
from lib.cross_net import CrossSparseAggrNet_v2


def l2norm(X, dim, eps=1e-8):
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    return torch.div(X, norm)


class AttentionPooling(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.attn_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x, mask=None):
        scores = self.attn_net(x)
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(-1) == 0, -1e9)
        attn_weights = F.softmax(scores, dim=1)
        return torch.sum(attn_weights * x, dim=1)

class MultiHeadAttentionPooling(nn.Module):
    """Aggregate token features with multiple attention heads."""
    def __init__(self, input_dim, num_heads=4):
        super().__init__()
        self.heads = nn.ModuleList([
            AttentionPooling(input_dim, input_dim // 2)
            for _ in range(num_heads)
        ])
        self.proj = nn.Sequential(
            nn.Linear(input_dim * num_heads, input_dim),
            nn.LayerNorm(input_dim),
        )

    def forward(self, x, mask=None):
        head_outs = [h(x, mask) for h in self.heads]
        return self.proj(torch.cat(head_outs, dim=-1))

class ResidualComplementaryFusion(nn.Module):
    """Fuse fine- and coarse-grained similarities with learnable residual scaling."""
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temp = temperature
        self.residual_scale = nn.Parameter(torch.tensor(0.5))
    
    def forward(self, fine_sims, coarse_sims):
        residual = fine_sims - coarse_sims
        
        scale = torch.sigmoid(self.residual_scale)
        
        fused_sims = fine_sims + scale * residual
        
        return fused_sims


class BidirectionalDistillationLoss(nn.Module):
    def __init__(self, temp_f2c=4.0, temp_c2f=4.0, weight_f2c=1.0, weight_c2f=0.5):
        super().__init__()
        self.temp_f2c = temp_f2c
        self.temp_c2f = temp_c2f
        self.weight_f2c = weight_f2c
        self.weight_c2f = weight_c2f
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')
    
    def forward(self, fine_sims, coarse_sims, return_breakdown=False):
        # Fine -> Coarse
        fine_p_i2t = F.softmax(fine_sims / self.temp_f2c, dim=1)
        coarse_logp_i2t = F.log_softmax(coarse_sims / self.temp_f2c, dim=1)
        loss_f2c_i2t = self.kl_loss(coarse_logp_i2t, fine_p_i2t)
        
        fine_p_t2i = F.softmax(fine_sims.t() / self.temp_f2c, dim=1)
        coarse_logp_t2i = F.log_softmax(coarse_sims.t() / self.temp_f2c, dim=1)
        loss_f2c_t2i = self.kl_loss(coarse_logp_t2i, fine_p_t2i)
        
        loss_f2c = loss_f2c_i2t + loss_f2c_t2i
        
        # Coarse -> Fine
        coarse_p_i2t = F.softmax(coarse_sims / self.temp_c2f, dim=1)
        fine_logp_i2t = F.log_softmax(fine_sims / self.temp_c2f, dim=1)
        loss_c2f_i2t = self.kl_loss(fine_logp_i2t, coarse_p_i2t)
        
        coarse_p_t2i = F.softmax(coarse_sims.t() / self.temp_c2f, dim=1)
        fine_logp_t2i = F.log_softmax(fine_sims.t() / self.temp_c2f, dim=1)
        loss_c2f_t2i = self.kl_loss(fine_logp_t2i, coarse_p_t2i)
        
        loss_c2f = loss_c2f_i2t + loss_c2f_t2i
        
        total_loss = self.weight_f2c * loss_f2c + self.weight_c2f * loss_c2f
        
        if return_breakdown:
            return total_loss, {
                'f2c_total': loss_f2c, 'c2f_total': loss_c2f
            }
        return total_loss


class VSEModel(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        
        self.img_enc = get_image_encoder(opt)
        self.txt_enc = get_text_encoder(opt)
        self.criterion = loss_select(opt, loss_type=opt.loss)
        self.Eiters = 0
        
        self.cross_net = CrossSparseAggrNet_v2(opt)
        
        # Coarse-grained multi-head attention pooling
        num_pool_heads = getattr(opt, 'num_pool_heads', 4)
        self.img_pool = MultiHeadAttentionPooling(opt.embed_size, num_heads=num_pool_heads)
        self.txt_pool = MultiHeadAttentionPooling(opt.embed_size, num_heads=num_pool_heads)
        
        self.fusion = ResidualComplementaryFusion()
        
        self.distill_loss_fn = BidirectionalDistillationLoss(
            temp_f2c=getattr(opt, 'distill_temp_f2c', 4.0),
            temp_c2f=getattr(opt, 'distill_temp_c2f', 4.0),
            weight_f2c=getattr(opt, 'distill_weight_f2c', 0.5),
            weight_c2f=getattr(opt, 'distill_weight_c2f', 0)
        )
        
        self.distill_start_epoch = getattr(opt, 'distill_start_epoch', 1)
        self.current_epoch = 0

    def set_epoch(self, epoch):
        """Set the current epoch from the training loop."""
        self.current_epoch = epoch

    def freeze_backbone(self):
        self.img_enc.freeze_backbone()
        self.txt_enc.freeze_backbone()

    def unfreeze_backbone(self):
        self.img_enc.unfreeze_backbone()
        self.txt_enc.unfreeze_backbone()

    def set_max_violation(self, max_violation=True):
        if max_violation:
            self.criterion.max_violation_on()
        else:
            self.criterion.max_violation_off()

    def forward_emb(self, images, captions, lengths):
        images = images.cuda()
        captions = captions.cuda()
        lengths = lengths.cuda()
        
        img_emb = self.img_enc(images)
        cap_emb = self.txt_enc(captions, lengths)
        return img_emb, cap_emb, lengths

    def _compute_coarse_sims(self, img_embs, cap_embs, cap_lens):
        img_global = l2norm(self.img_pool(img_embs), -1)
        
        N_txt, L, _ = cap_embs.shape
        device = cap_embs.device
        mask = torch.arange(L, device=device).expand(N_txt, L) < cap_lens.unsqueeze(1)
        txt_global = l2norm(self.txt_pool(cap_embs, mask), -1)
        
        return img_global.mm(txt_global.t())

    def forward_sim(self, img_embs, cap_embs, cap_lens):
        
        fine_sims, score_mask = self.cross_net(img_embs, cap_embs, cap_lens)
        
        coarse_sims = self._compute_coarse_sims(img_embs, cap_embs, cap_lens)

        fused_sims = self.fusion(fine_sims, coarse_sims)
        
        return {
            'fused': fused_sims,
            'fine': fine_sims,
            'coarse': coarse_sims,
            'score_mask': score_mask,
        }

    def forward(self, images, captions, lengths, img_ids=None, warmup_alpha=1.):
        self.Eiters += 1
        
        img_emb = self.img_enc(images)
        cap_emb = self.txt_enc(captions, lengths)
        
        if self.opt.multi_gpu:
            lengths = utils.concat_all_gather(lengths, keep_grad=False)
            img_ids = utils.concat_all_gather(img_ids, keep_grad=False)
            
            max_len = int(lengths.max())
            if max_len > cap_emb.shape[1]:
                pad = torch.zeros(cap_emb.shape[0], max_len - cap_emb.shape[1],
                                  cap_emb.shape[2], device=cap_emb.device)
                cap_emb = torch.cat([cap_emb, pad], dim=1)
            
            img_emb = utils.all_gather_with_grad(img_emb)
            cap_emb = utils.all_gather_with_grad(cap_emb)
        
        sim_dict = self.forward_sim(img_emb, cap_emb, lengths)
        
        fused_sims = sim_dict['fused']
        fine_sims = sim_dict['fine']
        coarse_sims = sim_dict['coarse']
        score_mask = sim_dict['score_mask']
        

        align_loss = self.criterion(img_emb, cap_emb, img_ids, fused_sims) * warmup_alpha

        ratio_loss = (score_mask.mean() - self.opt.sparse_ratio) ** 2

        distill_weight = getattr(self.opt, 'distill_weight', 1.0)

        if self.current_epoch < self.distill_start_epoch:
            distill_loss = torch.tensor(0.0, device=fused_sims.device)
        else:
            distill_loss = self.distill_loss_fn(fine_sims, coarse_sims)

        loss = align_loss + self.opt.aggr_ratio * ratio_loss + distill_weight * distill_loss
        
        return loss

    def get_loss_breakdown(self, images, captions, lengths, img_ids=None, warmup_alpha=1.):
        """Return the loss breakdown for logging or debugging."""
        self.Eiters += 1
        
        img_emb = self.img_enc(images)
        cap_emb = self.txt_enc(captions, lengths)
        
        if self.opt.multi_gpu:
            lengths = utils.concat_all_gather(lengths, keep_grad=False)
            img_ids = utils.concat_all_gather(img_ids, keep_grad=False)
            max_len = int(lengths.max())
            if max_len > cap_emb.shape[1]:
                pad = torch.zeros(cap_emb.shape[0], max_len - cap_emb.shape[1],
                                  cap_emb.shape[2], device=cap_emb.device)
                cap_emb = torch.cat([cap_emb, pad], dim=1)
            img_emb = utils.all_gather_with_grad(img_emb)
            cap_emb = utils.all_gather_with_grad(cap_emb)
        
        sim_dict = self.forward_sim(img_emb, cap_emb, lengths)
        
        fused_sims = sim_dict['fused']
        fine_sims = sim_dict['fine']
        coarse_sims = sim_dict['coarse']
        score_mask = sim_dict['score_mask']

        align_loss = self.criterion(img_emb, cap_emb, img_ids, fused_sims) * warmup_alpha
        ratio_loss = (score_mask.mean() - self.opt.sparse_ratio) ** 2
        
        if self.current_epoch < self.distill_start_epoch:
            distill_loss = torch.tensor(0.0, device=fused_sims.device)
            distill_breakdown = {'f2c_total': 0.0, 'c2f_total': 0.0}
        else:
            distill_loss, distill_breakdown = self.distill_loss_fn(
                fine_sims, coarse_sims, return_breakdown=True
            )
        
        distill_weight = getattr(self.opt, 'distill_weight', 0.01)
        total = align_loss + self.opt.aggr_ratio * ratio_loss + distill_weight * distill_loss
        
        return {
            'total': total,
            'align': align_loss,
            'ratio': ratio_loss,
            'distill': distill_loss,
            'distill_f2c': distill_breakdown['f2c_total'],
            'distill_c2f': distill_breakdown['c2f_total'],
            'sparse_ratio': score_mask.mean(),
            'warmup_alpha': warmup_alpha
        }


def create_optimizer(opt, model):
    decay_factor = 1e-4
    
    all_text_params = list(model.txt_enc.parameters())
    bert_params = list(model.txt_enc.bert.parameters())
    bert_params_ptr = [p.data_ptr() for p in bert_params]
    text_params_no_bert = [p for p in all_text_params if p.data_ptr() not in bert_params_ptr]
    
    params_list = [
        {'params': text_params_no_bert, 'lr': opt.learning_rate},
        {'params': bert_params, 'lr': opt.learning_rate * 0.1},
        {'params': model.img_enc.visual_encoder.parameters(), 'lr': opt.learning_rate * 0.1},
        {'params': model.img_enc.vision_proj.parameters(), 'lr': opt.learning_rate},
        {'params': model.cross_net.parameters(), 'lr': opt.learning_rate},
        {'params': model.criterion.parameters(), 'lr': opt.learning_rate},
        {'params': model.img_pool.parameters(), 'lr': opt.learning_rate},
        {'params': model.txt_pool.parameters(), 'lr': opt.learning_rate},
        {'params': model.fusion.parameters(), 'lr': opt.learning_rate},
    ]
    
    return torch.optim.AdamW(params_list, lr=opt.learning_rate, weight_decay=decay_factor)