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
    with torch.no_grad():
        for batch in sampler:
            x = transform(torch.cat([trajs.to(opt.device)[(batch[0], batch[1][i])] for i in range(opt.seq_len)], dim=1).float().to(opt.device))

            ent_map = model(x)
            ent_production = torch.mean(ent_map.reshape(x.shape[0], -1), dim = 1)
            entropy = ent_production.cpu().squeeze().numpy()
            ret.append(entropy)
            maps.append(ent_map.cpu().squeeze().numpy())

            loss += (- (torch.exp(opt.alpha * ent_production) - 1) / opt.alpha + (torch.exp(-(1 + opt.alpha) * ent_production) - 1) / (1 + opt.alpha)).sum().cpu().item()
    loss = loss / sampler.size


    ret = np.concatenate(ret)
    ret = ret.reshape(trajs.shape[0], -1)

    maps = np.concatenate(maps)
    maps = maps.reshape((-1, *opt.input_shape))
    return ret, maps, loss