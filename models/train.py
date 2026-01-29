import torch

#
# Training Algorithm
#

def train(opt, model, optim, trajs, sampler, transform):
    model.train()
    batch = next(sampler)

    x = transform(
        torch.cat([trajs.to(opt.device)[(batch[0], batch[1][i])] for i in range(opt.seq_len)], dim=1).float().to(
            opt.device))

    delta = x[:, 0, :, :] - x[:, 1, :, :]

    # core variables
    delta = delta.reshape(x.shape[0], 1, x.shape[2], x.shape[3])
    mapp = model(x)
    ent_production = torch.mean(mapp.reshape(x.shape[0], -1), dim=1)

    # regularization term
    R = opt.lam * torch.mean(
        torch.abs(mapp)
        * (1 - torch.heaviside(torch.abs(delta) - opt.threshold, torch.abs(delta) - opt.threshold)))

    optim.zero_grad()

    # alpha-NEEP loss
    loss = (- (torch.exp(opt.alpha * ent_production) - 1) / opt.alpha
            + (torch.exp(-(1 + opt.alpha) * ent_production) - 1) / (1 + opt.alpha)).mean()
    (loss + R).backward()
    optim.step()
    return loss.item(), R.item()