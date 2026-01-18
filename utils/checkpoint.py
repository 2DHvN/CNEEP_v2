import os
import shutil
import torch


def save_checkpoint(state, is_best, path):
    filename = os.path.join(path, "checkpoint.pth.tar")
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(path, "model_best.pth.tar"))


def load_checkpoint(opt, model, optimizer):
    if os.path.isfile(opt.resume):
        print("=> loading checkpoint '{}'".format(opt.resume))
        checkpoint = torch.load(opt.resume)
        opt.start = checkpoint["start"]
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        print(
            "=> loaded checkpoint '{}' (iteration {})".format(
                opt.resume, checkpoint["iteration"]
            )
        )
    else:
        print("=> no checkpoint found at '{}'".format(opt.resume))

