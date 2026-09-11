"""GPU impulse repair + frequency-dependent spectral reduction. Open in Thonny; press F5.

V5 uses numpy/scipy/soundfile and CUDA torch. No new dependencies.
GPU: impulse analysis and FFTs. CPU: local prediction fits and noise statistics.
This is an experimental restoration heuristic, not a trained audio model.
"""
from pathlib import Path
import json
import time
import numpy as np

INPUT_FILE = Path(r"C:\Users\carlo\Downloads\Love One Another.mp3")
MODE = "preview"   # preview offers choices; full uses PRESET immediately
PRESET = "saved"   # saved, gentle, balanced, strong
PREVIEW_START_SECONDS = 85.0
PRESETS = {"gentle": {"strength": 0.45}, "balanced": {"strength": 0.65}, "strong": {"strength": 0.82}}
USE_GPU = True
CLICK_THRESHOLD = 4.0       # Higher protects more transients; try 6 if drums soften
MAX_REPAIR_MS = 1.5
RUMBLE_CUTOFF_HZ = 20.0     # 0 disables


def moving_mean(x, width, backend):
    """Centered box average; double cumulative sums avoid cancellation drift."""
    half = width//2
    if backend == "numpy":
        padded = np.pad(x, ((0, 0), (half, half)), mode="edge")
        sums = np.concatenate([np.zeros((x.shape[0], 1)),
                               np.cumsum(padded, axis=-1, dtype=np.float64)], axis=-1)
    else:
        import torch
        import torch.nn.functional as F
        padded = F.pad(x.unsqueeze(0), (half, half), mode="replicate")[0]
        sums = torch.cat([torch.zeros_like(padded[:, :1], dtype=torch.float64),
                          torch.cumsum(padded, dim=-1, dtype=torch.float64)], dim=-1)
    return (sums[:, width:] - sums[:, :-width])/width


def impulse_scores(audio, sr, use_gpu):
    """Compare multi-scale curvature against a locally clipped noise envelope.

    Unlike amplitude outlier detection, this discounts smooth bass/vocal waves.
    Independent channels avoid stereo phase cancellation. Both backends implement
    the same operations; GPU uses bounded blocks rather than a whole-track STFT.
    """
    backend = "torch" if use_gpu else "numpy"
    if use_gpu:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable. Run the GPU installer or set USE_GPU=False.")
        print("Impulse analysis GPU:", torch.cuda.get_device_name(0), flush=True)
    block = sr*5
    context = max(64, round(sr*0.06))
    envelope_width = max(3, round(sr*0.02) | 1)
    lags = sorted(set(max(1, round(sr*ms/1000)) for ms in (0.04, 0.16, 0.5)))
    scores = np.zeros_like(audio)
    for begin in range(0, audio.shape[1], block):
        end = min(audio.shape[1], begin+block)
        left, right = max(0, begin-context), min(audio.shape[1], end+context)
        part = audio[:, left:right]
        if use_gpu:
            import torch.nn.functional as F
            x = torch.as_tensor(part, device="cuda", dtype=torch.float32)
            score = torch.zeros_like(x)
        else:
            x = part
            score = np.zeros_like(x)
        for lag in lags:
            if use_gpu:
                padded = F.pad(x.unsqueeze(0), (lag, lag), mode="replicate")[0]
                innovation = torch.abs(x - 0.5*(padded[:, :-2*lag]+padded[:, 2*lag:]))
                rough = moving_mean(innovation, envelope_width, backend)
                clipped = torch.minimum(innovation, 3*rough)
                scale = moving_mean(clipped, envelope_width, backend)
                ratio = innovation/(1.2533*scale.clamp_min(1e-6))
                score = torch.maximum(score, ratio.to(torch.float32))
            else:
                padded = np.pad(x, ((0, 0), (lag, lag)), mode="edge")
                innovation = np.abs(x - 0.5*(padded[:, :-2*lag]+padded[:, 2*lag:]))
                rough = moving_mean(innovation, envelope_width, backend)
                clipped = np.minimum(innovation, 3*rough)
                scale = moving_mean(clipped, envelope_width, backend)
                ratio = innovation/(1.2533*np.maximum(scale, 1e-6))
                score = np.maximum(score, ratio).astype(np.float32)
        if use_gpu:
            score = score.cpu().numpy()
        scores[:, begin:end] = score[:, begin-left:end-left]
        print(f"Impulse analysis: {end/audio.shape[1]:.0%}", flush=True)
    return scores


def reconstruct_v4(audio, scores, sr, threshold, max_ms):
    """Grow high-confidence seeds to weaker shoulders and repair narrow spans."""
    if threshold <= 0 or max_ms <= 0:
        raise ValueError("Click threshold and max repair length must be positive.")
    result = audio.copy()
    max_samples = max(1, round(sr*max_ms/1000))
    pad = max(1, round(sr*0.04/1000))
    stats = []
    for ch, score in enumerate(scores):
        seeds = score > threshold
        support = score > threshold*0.5
        edges = np.diff(np.r_[False, support, False].astype(np.int8))
        intervals = []
        for start, end in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
            if not seeds[start:end].any():
                continue
            start, end = max(1, start-pad), min(audio.shape[1]-1, end+pad)
            if end <= start:
                continue
            if intervals and start <= intervals[-1][1]:
                intervals[-1] = (intervals[-1][0], max(end, intervals[-1][1]))
            else:
                intervals.append((start, end))
        count = changed = 0
        for start, end in intervals:
            if end-start > max_samples:
                continue
            result[ch, start:end] = np.linspace(audio[ch, start-1], audio[ch, end],
                                                 end-start+2)[1:-1]
            count += 1
            changed += end-start
        stats.append((count, changed))
        print(f"Channel {ch+1}: {count} regions repaired; "
              f"{100*changed/audio.shape[1]:.3f}% of samples", flush=True)
    return result, stats


def spectral_mask_v4(magnitude, sr, n_fft, hop, amount):
    """Shared stereo gain; local lower-percentile noise estimate is heuristic."""
    from scipy.ndimage import percentile_filter, median_filter, gaussian_filter
    frames = magnitude.shape[1]
    step = max(1, round(0.12*sr/hop))
    grid = np.arange(0, frames, step)
    coarse = np.stack([np.mean(magnitude[:, i:min(i+step, frames)]**2, axis=1)
                       for i in grid], axis=1)
    width = max(3, round(6.0*sr/(hop*step)) | 1)
    noise_coarse = percentile_filter(coarse, percentile=20,
                                     size=(1, width), mode='nearest')
    noise = np.stack([np.interp(np.arange(frames), grid, row) for row in noise_coarse])
    power = magnitude.astype(np.float64)**2
    estimate = np.sqrt(np.maximum(0, 1 - 1.5*noise/np.maximum(power, 1e-20)))
    frequencies = np.arange(magnitude.shape[0])*sr/n_fft
    max_db = amount*np.interp(frequencies, [0, 100, 500, 2500, 7000, sr/2],
                              [1, 2, 7, 12, 18, 18])
    floor = 10**(-max_db[:, None]/20)
    gain = floor + (1-floor)*estimate
    # Protect prominent tones, relative to surrounding frequency bins.
    local_spectrum = median_filter(magnitude, size=(17, 1), mode='nearest')
    prominence = 20*np.log10(np.maximum(magnitude, 1e-10)/np.maximum(local_spectrum, 1e-10))
    protect = np.clip((prominence-6)/18, 0, 0.85)
    # Protect abrupt broadband attacks relative to preceding 100 ms.
    energy = np.mean(power, axis=0)
    lag = max(1, round(0.1*sr/hop))
    prior = np.r_[np.full(lag, energy[0]), energy][:-lag]
    attack = np.clip((energy/np.maximum(prior, 1e-15)-2)/6, 0, 0.7)
    protect = np.maximum(protect, attack[None, :])
    gain = gain + (1-gain)*protect
    gain = gaussian_filter(gain, sigma=(1.0, max(0.5, 0.018*sr/hop)), mode='nearest')
    return np.clip(gain, floor, 1).astype(np.float32)


def adaptive_reduce(audio, sr, amount, use_gpu, detail_protection=True):
    """GPU FFTs with CPU noise statistics; linked stereo and overlapping blocks.

    This estimates a noise floor, not a known clean/noise separation. The local
    estimate can include music. Tone/attack protection reduces but cannot remove
    that ambiguity. Overlaps crossfade independently estimated block boundaries.
    """
    from scipy.signal import stft, istft, get_window
    if not 0 <= amount <= 1:
        raise ValueError('BACKGROUND_REDUCTION must be between 0 and 1.')
    if amount == 0:
        return audio.copy()
    n_fft = 2**int(round(np.log2(sr*0.085)))
    hop = n_fft//4
    window = get_window('hann', n_fft).astype(np.float32)
    if use_gpu:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA is unavailable.')
        gpu_window = torch.as_tensor(window, device='cuda')
    block, overlap, context = sr*20, sr, sr*4
    output = np.zeros_like(audio)
    weight = np.zeros(audio.shape[1], dtype=np.float32)
    for begin in range(0, audio.shape[1], block-overlap):
        end = min(audio.shape[1], begin+block)
        left, right = max(0, begin-context), min(audio.shape[1], end+context)
        chunk = audio[:, left:right]
        length = chunk.shape[1]
        chunk = np.pad(chunk, ((0, 0), (0, (-length) % hop)))
        if use_gpu:
            tensor = torch.as_tensor(chunk, device='cuda', dtype=torch.float32)
            z = torch.stft(tensor, n_fft=n_fft, hop_length=hop, window=gpu_window,
                           center=True, pad_mode='constant', return_complex=True)
            magnitude = torch.sqrt(torch.mean(torch.abs(z)**2, dim=0)).cpu().numpy()
        else:
            _, _, z = stft(chunk, fs=sr, window=window, nperseg=n_fft,
                           noverlap=n_fft-hop, boundary='zeros', padded=False, axis=-1)
            z *= window.sum()
            magnitude = np.sqrt(np.mean(np.abs(z)**2, axis=0))
        gain = spectral_mask(magnitude, sr, n_fft, hop, amount, detail_protection)
        if use_gpu:
            z *= torch.as_tensor(gain, device='cuda')[None, :, :]
            reconstructed = torch.istft(z, n_fft=n_fft, hop_length=hop,
                                         window=gpu_window, center=True,
                                         length=chunk.shape[1]).cpu().numpy()
        else:
            z *= gain[None, :, :]
            _, reconstructed = istft(z/window.sum(), fs=sr, window=window,
                                     nperseg=n_fft, noverlap=n_fft-hop,
                                     boundary=True, time_axis=-1, freq_axis=-2)
        section = reconstructed[:, begin-left:end-left]
        ramp = np.ones(end-begin, dtype=np.float32)
        fade = min(overlap, len(ramp))
        if begin > 0:
            ramp[:fade] *= np.linspace(0, 1, fade, dtype=np.float32)
        if end < audio.shape[1]:
            ramp[-fade:] *= np.linspace(1, 0, fade, dtype=np.float32)
        output[:, begin:end] += section*ramp
        weight[begin:end] += ramp
        print(f'Background reduction: {end/audio.shape[1]:.0%}', flush=True)
        if end == audio.shape[1]:
            break
    output /= np.maximum(weight, 1e-10)
    if output.shape != audio.shape or not np.isfinite(output).all():
        raise RuntimeError('Background reducer returned invalid audio.')
    return output


def ar_forecast(context, count, order=24):
    """Ridge-regularized autoregression; return None if extrapolation is unstable."""
    from scipy.linalg import solve
    from numpy.lib.stride_tricks import sliding_window_view
    context = np.asarray(context, dtype=np.float64)
    if len(context) < order*4 or not np.isfinite(context).all():
        return None
    mean = context.mean()
    scale = max(float(np.std(context)), 1e-8)
    x = (context-mean)/scale
    windows = sliding_window_view(x, order+1)
    design, target = windows[:, :-1].copy(), windows[:, -1]
    gram = design.T @ design
    gram.flat[::order+1] += max(1e-8, 0.001*np.trace(gram)/order)
    try:
        coefficients = solve(gram, design.T @ target, assume_a='pos')
    except (np.linalg.LinAlgError, ValueError):
        return None
    buffer = np.empty(order+count)
    buffer[:order] = x[-order:]
    for k in range(count):
        value = float(buffer[k:k+order] @ coefficients)
        if not np.isfinite(value) or abs(value) > max(12., 3*np.max(np.abs(x))):
            return None
        buffer[order+k] = value
    return buffer[order:]*scale+mean


def predict_gap(signal, start, end, sr):
    """Check held-out neighbors, then blend left and right waveform predictions."""
    width = end-start
    context_size = max(128, round(sr*0.008))
    if start < context_size or end+context_size > len(signal):
        return None
    left = signal[start-context_size:start]
    right = signal[end:end+context_size][::-1]
    horizon = min(48, max(8, width))
    for context in (left, right):
        validation = ar_forecast(context[:-horizon], horizon)
        if validation is None:
            return None
        error = np.sqrt(np.mean((validation-context[-horizon:])**2))
        if error > 0.55*max(float(np.std(context)), 1e-7):
            return None
    forward, backward = ar_forecast(left, width), ar_forecast(right, width)
    if forward is None or backward is None:
        return None
    backward = backward[::-1]
    neighborhood = np.r_[left, right]
    if np.sqrt(np.mean((forward-backward)**2)) > 0.8*max(np.std(neighborhood), 1e-7):
        return None
    weight = np.linspace(0, 1, width+2)[1:-1]
    prediction = (1-weight)*forward + weight*backward
    if np.max(np.abs(prediction)) > 1.5*max(np.max(np.abs(neighborhood)), 1e-7):
        return None
    return prediction


def reconstruct(audio, scores, sr, threshold=4., max_ms=1.5):
    # Reuse exactly the V4 candidate selection; repair method is the changed variable.
    linear, counts = reconstruct_v4(audio, scores, sr, threshold, max_ms)
    predicted = linear.copy()
    stats = []
    for ch in range(audio.shape[0]):
        # Mask differences identify only samples actually changed by the V4 repair.
        mask = linear[ch] != audio[ch]
        edges = np.diff(np.r_[False, mask, False].astype(np.int8))
        ar_count = fallback = 0
        for start, end in zip(np.flatnonzero(edges==1), np.flatnonzero(edges==-1)):
            estimate = predict_gap(linear[ch], start, end, sr)
            if estimate is None:
                fallback += 1
            else:
                predicted[ch, start:end] = estimate
                ar_count += 1
        stats.append(dict(candidate_regions=int(counts[ch][0]), repaired_samples=int(counts[ch][1]),
                          predictive_regions=ar_count, linear_fallback_regions=fallback))
        print(f'Channel {ch+1}: predictive {ar_count}, linear fallback {fallback}', flush=True)
    return predicted, stats


def spectral_mask(magnitude, sr, n_fft, hop, amount, detail_protection=True):
    baseline = spectral_mask_v4(magnitude, sr, n_fft, hop, amount)
    if not detail_protection:
        return baseline
    from scipy.ndimage import gaussian_filter1d, median_filter
    # Smoothly ease attenuation in louder passages, relative to this block.
    power = magnitude.astype(np.float64)**2
    energy = np.mean(power, axis=0)
    smoothed = gaussian_filter1d(energy, max(1., 0.3*sr/hop), mode='nearest')
    low, high = np.percentile(smoothed, [20, 80])
    activity = np.clip((smoothed-low)/max(high-low, 1e-15), 0, 1)
    # Protect sustained narrow spectral peaks. This is a heuristic, not vocal isolation.
    neighborhood = median_filter(magnitude, size=(17, 1), mode='nearest')
    prominent = np.clip((magnitude/np.maximum(neighborhood,1e-10)-2)/6, 0, 1)
    sustained = gaussian_filter1d(prominent, max(1.,0.08*sr/hop), axis=-1, mode='nearest')
    protection = np.maximum(0.12*activity[None,:], 0.20*sustained)
    return (baseline+(1-baseline)*protection).astype(np.float32)


def safe_output(path, source):
    path, source = Path(path), Path(source)
    if path.resolve() == source.resolve() or (path.exists() and path.samefile(source)):
        raise ValueError('Output would overwrite the original input.')
    return path


def load_preset(path):
    if not path.exists():
        return 'balanced'
    try:
        data = json.loads(path.read_text(encoding='utf-8'))
        name = data['selected_preset']
        if data.get('version') != 5 or name not in PRESETS:
            raise ValueError('Unknown preset or version')
        return name
    except (ValueError, KeyError, TypeError):
        print('Saved preset is invalid; using Balanced.')
        return 'balanced'


def save_preset(path, source, name):
    safe_output(path, source)
    data = dict(version=5, input=str(source.resolve()), selected_preset=name,
                settings=PRESETS[name], click_threshold=CLICK_THRESHOLD,
                max_repair_ms=MAX_REPAIR_MS, rumble_cutoff_hz=RUMBLE_CUTOFF_HZ)
    path.write_text(json.dumps(data, indent=2), encoding='utf-8')


def matched_previews(variants, sr):
    """Match active midband RMS, capped at +/-2 dB, then apply shared headroom.

    This is an approximate music-level comparison, not integrated LUFS matching.
    Matching is only for previews. Full masters preserve gain unless peak-limited.
    """
    from scipy.signal import butter, sosfiltfilt
    reference = variants['Original']
    upper = min(3000, sr*0.4)
    sos = butter(2, [min(300, upper/3), upper], btype='bandpass', fs=sr, output='sos')
    band = sosfiltfilt(sos, reference, axis=-1)
    block = max(1, sr//20)
    count = band.shape[1]//block
    levels = np.array([np.mean(band[:,i*block:(i+1)*block]**2) for i in range(count)])
    active = np.repeat(levels >= np.percentile(levels,40), block)
    active = np.pad(active,(0,reference.shape[1]-len(active)),constant_values=False)
    ref_level = np.sqrt(np.mean(band[:,active]**2))
    adjusted, gains = {}, {}
    for name, audio in variants.items():
        filtered = sosfiltfilt(sos,audio,axis=-1)
        level = np.sqrt(np.mean(filtered[:,active]**2))
        gain = float(np.clip(ref_level/max(level,1e-12),10**(-2/20),10**(2/20))) if ref_level>1e-10 else 1.
        adjusted[name] = audio*gain
        gains[name] = gain
    peak = max(float(np.max(np.abs(a))) for a in adjusted.values())
    headroom = min(1.,.98/max(peak,1e-10))
    return {k:(v*headroom).astype(np.float32) for k,v in adjusted.items()}, dict(gains=gains,shared_headroom=headroom)


def create_previews(original, sr, output_dir, source, use_gpu, start_seconds=85.):
    import soundfile as sf
    from scipy.signal import butter, sosfiltfilt
    length = original.shape[1]
    start = max(0, min(round(start_seconds*sr), length-sr*15))
    end = min(length,start+sr*15)
    # Context lets the local detector and spectral estimator see outside the preview.
    left,right=max(0,start-sr*4),min(length,end+sr*4)
    raw=original[:,left:right]
    filtered=raw.copy()
    if RUMBLE_CUTOFF_HZ:
        sos=butter(4,RUMBLE_CUTOFF_HZ,btype='highpass',fs=sr,output='sos')
        filtered=sosfiltfilt(sos,raw,axis=-1).astype(np.float32)
    scores=impulse_scores(filtered,sr,use_gpu)
    repaired,stats=reconstruct(filtered,scores,sr,CLICK_THRESHOLD,MAX_REPAIR_MS)
    legacy,_=reconstruct_v4(filtered,scores,sr,CLICK_THRESHOLD,MAX_REPAIR_MS)
    crop=slice(start-left,end-left)
    variants={'Original':raw[:,crop]}
    variants['V4 Reference']=adaptive_reduce(legacy,sr,.65,use_gpu,detail_protection=False)[:,crop]
    for name,settings in PRESETS.items():
        variants[name.title()]=adaptive_reduce(repaired,sr,settings['strength'],use_gpu)[:,crop]
    adjusted,matching=matched_previews(variants,sr)
    output_dir.mkdir(parents=True,exist_ok=True)
    for name,audio in adjusted.items():
        path=safe_output(output_dir/f'{source.stem} - {name} Preview.wav',source)
        sf.write(str(path),audio.T,sr,subtype='PCM_24')
    report=dict(start_seconds=start/sr,end_seconds=end/sr,matching=matching,
                click_repair=stats,presets=PRESETS,
                note='Approximate active 300-3000 Hz RMS matching; not LUFS. V4 reference uses strength 0.65.')
    (output_dir/'Preview report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(f'\nPREVIEWS READY: {output_dir}\nPassage: {start/sr:.1f}-{end/sr:.1f} seconds',flush=True)
    return report


def restore_full(original,sr,source,output_dir,name,use_gpu):
    import soundfile as sf
    from scipy.signal import butter,sosfiltfilt
    started=time.perf_counter()
    strength=PRESETS[name]['strength']
    filtered=original.copy()
    if RUMBLE_CUTOFF_HZ:
        sos=butter(4,RUMBLE_CUTOFF_HZ,btype='highpass',fs=sr,output='sos')
        filtered=sosfiltfilt(sos,original,axis=-1).astype(np.float32)
    scores=impulse_scores(filtered,sr,use_gpu)
    repaired,stats=reconstruct(filtered,scores,sr,CLICK_THRESHOLD,MAX_REPAIR_MS)
    del scores
    restored=adaptive_reduce(repaired,sr,strength,use_gpu)
    if restored.shape!=original.shape or not np.isfinite(restored).all():
        raise RuntimeError('Invalid processed audio; no audio files written.')
    output_dir.mkdir(parents=True,exist_ok=True)
    stem=source.stem+' - V5 '+name.title()
    peak=float(np.max(np.abs(restored)))
    gain=min(1.,.999/max(peak,1e-10))
    path=safe_output(output_dir/(stem+'.wav'),source)
    sf.write(str(path),(restored*gain).T,sr,subtype='PCM_24')
    for suffix,residual in [('Removed Clicks',filtered-repaired),('Removed Background',repaired-restored),('Removed Rumble',original-filtered)]:
        target=safe_output(output_dir/(stem+' - '+suffix+'.wav'),source)
        sf.write(str(target),residual.T,sr,subtype='FLOAT')
    report=dict(version=5,input=str(source),sample_rate=sr,samples=original.shape[1],
                preset=name,settings=PRESETS[name],gpu=use_gpu,click_repair=stats,
                master_gain=gain,elapsed_seconds=time.perf_counter()-started,
                note='Residuals before master gain; no normalization. Predictions validated on neighboring samples, not known clean audio.')
    (output_dir/(stem+' - Report.json')).write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(f'\nSAVED: {path}\nElapsed: {report["elapsed_seconds"]:.1f} s',flush=True)
    return report


def main():
    import soundfile as sf
    if MODE not in ('preview','full') or PRESET not in ('saved',*PRESETS):
        raise ValueError('MODE must be preview/full; PRESET must be saved/gentle/balanced/strong.')
    raw,sr=sf.read(str(INPUT_FILE),always_2d=True,dtype='float32')
    original=np.ascontiguousarray(raw.T)
    if original.shape[1]<sr or not np.isfinite(original).all():
        raise ValueError('Input must contain at least one second of finite audio.')
    if CLICK_THRESHOLD<=0 or MAX_REPAIR_MS<=0 or not 0<=RUMBLE_CUTOFF_HZ<sr/2:
        raise ValueError('Invalid click or rumble settings.')
    folder=INPUT_FILE.parent/(INPUT_FILE.stem+' - V5')
    preset_path=folder/'Selected preset.json'
    selected=load_preset(preset_path) if PRESET=='saved' else PRESET
    if MODE=='preview':
        create_previews(original,sr,folder,INPUT_FILE,USE_GPU,PREVIEW_START_SECONDS)
        print('Play the WAV previews in that folder, then return here.')
        answer=input(f'Choose g=Gentle, b=Balanced, s=Strong; Enter={selected}; q=stop: ').strip().lower()
        if answer=='q':
            print('Stopped after previews. Full recording was not processed.')
            return
        choices={'g':'gentle','b':'balanced','s':'strong','':selected}
        while answer not in choices:
            answer=input('Enter g, b, s, or q: ').strip().lower()
            if answer=='q':
                return
        selected=choices[answer]
    folder.mkdir(parents=True,exist_ok=True)
    save_preset(preset_path,INPUT_FILE,selected)
    restore_full(original,sr,INPUT_FILE,folder,selected,USE_GPU)


if __name__=='__main__':
    main()
