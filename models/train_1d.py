import torch

#
# Training Algorithm for 1D Models
#

def train_1d(opt, model, optim, trajs, sampler, transform):
    model.train()
    batch = next(sampler)

    # trajs shape: (M, L, 1, Lx)
    # x shape after transform and cat: (batch_size, seq_len * channels, Lx)
    x = transform(
        torch.cat([trajs[(batch[0], batch[1][i])].to(opt.device) for i in range(opt.seq_len)], dim=1).float().to(
            opt.device))

    # delta is the difference between sequential frames
    delta = x[:, 0, :] - x[:, 1, :]

    # core variables
    delta = delta.reshape(x.shape[0], 1, x.shape[2])
    mapp = model(x) / opt.scalar
    ent_production = torch.mean(mapp.reshape(x.shape[0], -1), dim=1)

    # regularization term
    R = opt.lam * torch.mean(
        torch.abs(mapp)
        * (1 - torch.heaviside(torch.abs(delta) - opt.threshold, torch.abs(delta) - opt.threshold)))

    optim.zero_grad()

    # alpha-NEEP loss
    if opt.alpha == 0:
        loss = (- ent_production + (torch.exp(-ent_production) - 1)).mean()
    else:
        loss = (- (torch.exp(opt.alpha * ent_production) - 1) / opt.alpha
            + (torch.exp(-(1 + opt.alpha) * ent_production) - 1) / (1 + opt.alpha)).mean()

    (loss * opt.loss_scalar + R).backward()

    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=opt.clip_norm)

    optim.step()
    return loss.item(), R.item()
