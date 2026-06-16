import torch
import numpy as np

#
# Validation
#

def validate(opt, model, trajs, sampler, transform):
    model.eval()

    ret = []
    maps = []
    loss = 0
    n_samples = 0
    with torch.no_grad():
        for batch in sampler:
            b0 = batch[0].to(trajs.device)
            slices = [trajs[(b0, batch[1][i].to(trajs.device))] for i in range(opt.seq_len)]
            x = transform(torch.cat(slices, dim=1).float().to(opt.device))

            ent_map = model(x) / opt.scalar
            dx = getattr(opt, 'dx', 1.0)
            vol = x.shape[2] * x.shape[3] * (dx ** 2)
            ep_density = torch.mean(ent_map.reshape(x.shape[0], -1), dim=1)
            ent_production = ep_density * vol
            entropy = ent_production.cpu().squeeze().numpy()
            ret.append(entropy)
            maps.append(ent_map.cpu().squeeze().numpy())

            if opt.alpha == 0:
                loss += (- ep_density + (torch.exp(-ep_density) - 1)).sum().cpu().item()
            else:
                loss += (- (torch.exp(opt.alpha * ep_density) - 1) / opt.alpha + (torch.exp(-(1 + opt.alpha) * ep_density) - 1) / (1 + opt.alpha)).sum().cpu().item()
            
            n_samples += x.shape[0]
            
    loss = loss / n_samples


    ret = np.concatenate(ret)
    ret = ret.reshape(trajs.shape[0], -1)

    maps = np.concatenate(maps)
    maps = maps.reshape((-1, *opt.input_shape))
    return ret, maps, loss