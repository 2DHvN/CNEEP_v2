"""Memory-bounded held-out field/EP maps and signed shell spectra.

Known maps use the model-defined finite-lag active-work convention. Learned maps
use the trained network's local work in the same observation interval. Neither
map is a claim of a unique local thermodynamic dissipation density.
"""
from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from tqdm.auto import tqdm

from .diagnostics import json_safe


def _positive_integer(value, name):
    if isinstance(value,bool) or not isinstance(value,(int,np.integer)) or value<1:
        raise ValueError(f"{name} must be a positive integer.")


def _shell_order(label):
    return (1,0) if label=='tail' else (0,int(label[1:]))


def analyze_test_fields(fit, test_fields, physical, frame_dt, *, lag_frames=1,
                        batch_size=32, video_trajectory=0, video_stride=5,
                        video_max_frames=200, show_progress=True, time_offset=0.0):
    """Stream fixed-lag pairs from every held-out replica into maps and spectra.

    Pair starts are 0, lag_frames, 2*lag_frames, ... as in
    ``evaluate_shell_force``. Each pair contributes to the mean/spectrum; only
    every ``video_stride``-th pair from one selected trajectory is retained for
    display. Changing video_stride never changes the physical EP denominator.

    ``time_offset`` is a scalar or one starting physical time per trajectory,
    e.g. the saved run's burn-in time. video_phi contains actual left-endpoint
    states. video_phi_midpoint and both EP maps refer to the interval midpoint.
    All rate maps have units of local dS/time: their spatial SUM is the extensive
    rate, with no additional N factor. Signed values are preserved.
    """
    for name,value in [('lag_frames',lag_frames),('batch_size',batch_size),
                       ('video_stride',video_stride),('video_max_frames',video_max_frames)]:
        _positive_integer(value,name)
    if not np.isfinite(frame_dt) or frame_dt<=0:
        raise ValueError('frame_dt must be positive and finite.')
    trajectories=[test_fields] if isinstance(test_fields,np.ndarray) else list(test_fields)
    if not trajectories:
        raise ValueError('At least one held-out trajectory is required.')
    if (isinstance(video_trajectory,bool) or not isinstance(video_trajectory,(int,np.integer))
            or not 0<=video_trajectory<len(trajectories)):
        raise ValueError('video_trajectory must identify an existing held-out trajectory.')
    offsets=np.asarray(time_offset,dtype=np.float64)
    if offsets.ndim==0:
        offsets=np.full(len(trajectories),float(offsets))
    if offsets.shape!=(len(trajectories),) or not np.isfinite(offsets).all():
        raise ValueError('time_offset must be finite and scalar or have one entry per trajectory.')
    pair_counts=[]
    for trajectory in trajectories:
        if (not isinstance(trajectory,np.ndarray) or trajectory.ndim!=4
                or trajectory.shape[1:]!=(1,*physical.shape) or len(trajectory)<=lag_frames):
            raise ValueError(f'Each trajectory must have shape [time,1,{physical.shape[0]},{physical.shape[1]}] and enough frames.')
        pair_counts.append(len(range(0,len(trajectory)-lag_frames,lag_frames)))
    total_pairs=sum(pair_counts)
    interval=float(frame_dt*lag_frames)
    model,normalizer=fit.model,fit.normalizer
    parameter=next(model.parameters(),None)
    if parameter is None:
        parameter=next(model.buffers(),None)
    device=parameter.device if parameter is not None else torch.device('cpu')
    dtype=parameter.dtype if parameter is not None else torch.float64
    if dtype not in (torch.float32,torch.float64):
        raise ValueError('The trained model must use float32 or float64 field parameters.')
    radius=int(model.max_distance)
    if radius<0:
        raise ValueError('Model max_distance must be nonnegative.')
    known_sum=np.zeros(physical.shape,dtype=np.float64)
    nn_sum=np.zeros(physical.shape,dtype=np.float64)
    channel_sum={}
    nn_spectrum_sum=None
    nn_labels=None
    video={key:[] for key in ('video_phi','video_phi_midpoint','video_known_ep_rate','video_nn_ep_rate',
                             'video_time_start','video_time_mid','video_time_end','video_pair_indices')}
    was_training=model.training
    started=perf_counter()
    model.eval()
    try:
        with torch.no_grad(),tqdm(total=total_pairs,desc='Held-out EP maps and spectrum',unit='pair',
                                 mininterval=.5,disable=not show_progress) as bar:
            for trajectory_index,trajectory in enumerate(trajectories):
                starts=range(0,len(trajectory)-lag_frames,lag_frames)
                for begin in range(0,len(starts),batch_size):
                    indices=np.asarray(starts[begin:begin+batch_size],dtype=np.int64)
                    # Only the current pair batch is copied/cast. np.memmap inputs
                    # remain on disk; the complete trajectory is never cast.
                    pairs=np.stack((trajectory[indices,0],trajectory[indices+lag_frames,0]),axis=1).astype(np.float64,copy=False)
                    if not np.isfinite(pairs).all():
                        raise ValueError(f'Nonfinite held-out pair in trajectory {trajectory_index}.')
                    z=normalizer.transform(torch.from_numpy(pairs)).to(device=device,dtype=dtype)
                    maps=model(z,return_maps=True)
                    if not isinstance(maps,torch.Tensor) or maps.ndim!=4 or maps.shape[0]!=len(indices) or tuple(maps.shape[-2:])!=physical.shape:
                        raise ValueError('return_maps=True must return [batch,branches,Ny,Nx].')
                    branch_maps=maps.detach().to(device='cpu',dtype=torch.float64).numpy()/interval
                    if not np.isfinite(branch_maps).all():
                        raise FloatingPointError('Nonfinite learned EP maps.')
                    if nn_spectrum_sum is None:
                        nn_spectrum_sum=np.zeros(branch_maps.shape[1],dtype=np.float64)
                        start_k=int(getattr(model,'start_k',0))
                        nn_labels=[f'r{start_k+j}' for j in range(branch_maps.shape[1])]
                    elif branch_maps.shape[1]!=len(nn_spectrum_sum):
                        raise ValueError('The model changed its branch count during analysis.')
                    nn_maps=branch_maps.sum(axis=1)
                    nn_sum+=nn_maps.sum(axis=0)
                    nn_spectrum_sum+=branch_maps.sum(axis=(0,2,3))
                    for local_index,start in enumerate(indices):
                        left,right=pairs[local_index]
                        midpoint=(left+right)/2
                        delta=right-left
                        known_map=np.zeros(physical.shape,dtype=np.float64)
                        channels=physical.active_channels(midpoint,max_distance=radius)
                        for label,nu in channels.items():
                            rate=-nu*delta/(physical.config.D*interval)
                            if not np.isfinite(rate).all():
                                raise FloatingPointError('Nonfinite known-model endpoint EP maps.')
                            known_map+=rate
                            channel_sum[label]=channel_sum.get(label,0.0)+float(rate.sum())
                        known_sum+=known_map
                        ordinal=begin+local_index
                        if (trajectory_index==video_trajectory and ordinal%video_stride==0
                                and len(video['video_phi'])<video_max_frames):
                            t0=float(offsets[trajectory_index]+int(start)*frame_dt)
                            t1=float(offsets[trajectory_index]+(int(start)+lag_frames)*frame_dt)
                            video['video_phi'].append(left.copy())
                            video['video_phi_midpoint'].append(midpoint.copy())
                            video['video_known_ep_rate'].append(known_map.copy())
                            video['video_nn_ep_rate'].append(nn_maps[local_index].copy())
                            video['video_time_start'].append(t0)
                            video['video_time_mid'].append((t0+t1)/2)
                            video['video_time_end'].append(t1)
                            video['video_pair_indices'].append(int(start))
                    bar.update(len(indices))
    finally:
        model.train(was_training)
    channel_labels=list(channel_sum)
    channel_rates=np.asarray([channel_sum[key]/total_pairs for key in channel_labels])
    shells={}
    for label,rate in zip(channel_labels,channel_rates):
        shell=label.split(':',1)[1]
        shells[shell]=shells.get(shell,0.0)+float(rate)
    known_labels=sorted(shells,key=_shell_order)
    result={key:np.asarray(values,dtype=np.int64 if key=='video_pair_indices' else np.float64)
            for key,values in video.items()}
    result.update(mean_known_ep_rate=known_sum/total_pairs,mean_nn_ep_rate=nn_sum/total_pairs,
                  nn_shell_labels=nn_labels,nn_shell_rates=nn_spectrum_sum/total_pairs,
                  known_shell_labels=known_labels,known_shell_rates=np.asarray([shells[key] for key in known_labels]),
                  known_channel_labels=channel_labels,known_channel_rates=channel_rates,
                  metadata={'model_id':physical.config.model_id,'n_sites':physical.N,'D':physical.config.D,
                      'total_pairs':total_pairs,'per_trajectory_pairs':pair_counts,'n_test_trajectories':len(trajectories),
                      'frame_dt':float(frame_dt),'lag_frames':int(lag_frames),'observation_interval':interval,
                      'summed_observed_duration':total_pairs*interval,
                      'pairing':'nonoverlapping fixed-lag intervals within each trajectory; no cross-replica pairs',
                      'averaging':'all held-out pairs weighted by their equal observation interval',
                      'video_trajectory':int(video_trajectory),'video_stride':int(video_stride),
                      'video_max_frames':int(video_max_frames),'video_frames':len(video['video_phi']),
                      'time_offsets':offsets.tolist(),'profile_time_coordinate':'left endpoint',
                      'ep_time_coordinate':'interval midpoint','video_stride_changes_lag':False,
                      'map_units':'local dS / time; spatial sum is extensive rate, without an extra N factor',
                      'spectrum_units':'signed extensive EP rate per shell; mechanisms summed for known-shell spectrum',
                      'known_map':'model-defined active work at the observation-pair midpoint, not streaming integration-step EPR',
                      'learned_map':'trained thermodynamic-force work; its local attribution can differ by a gradient gauge',
                      'normalization':{'mean':float(normalizer.mean),'std':float(normalizer.std),'rule':'fixed training affine map; d_z dz=d_phi dphi'},
                      'elapsed_seconds':perf_counter()-started})
    return result


def save_analysis_npz(analysis: dict, path) -> Path:
    """Save small visualization arrays and JSON metadata without pickled objects."""
    destination=Path(path)
    destination.parent.mkdir(parents=True,exist_ok=True)
    arrays={key:np.asarray(value) for key,value in analysis.items() if key!='metadata'}
    arrays['metadata_json']=np.asarray(json.dumps(json_safe(analysis.get('metadata',{})),ensure_ascii=False))
    np.savez_compressed(destination,**arrays)
    return destination if destination.suffix=='.npz' else Path(str(destination)+'.npz')
