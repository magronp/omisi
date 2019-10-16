#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

__author__ = 'Paul Magron -- IRIT' \
             'Original code by Kaituo XU -- https://github.com/kaituoxu/Conv-TasNet'
__docformat__ = 'reStructuredText'
__all__ = []


def get_separation_score(src_ref, src_est):
    """Calculate Scale-Invariant SDR improvement (SI-SDRi)
    Args:
        src_ref: numpy.ndarray (nsrc, nsamples) - ground truth sources
        src_est: numpy.ndarray (nsrc, nsamples) - estimated sources
    Returns:
        score: int
    """
    # Expected inputs of size (nsrc x nsampl)
    if src_ref.shape[0]>src_ref.shape[1]:
        src_ref = src_ref.T
        src_est = src_est.T
        
    # Calculate the mixture
    mix = np.sum(src_ref, axis=0)

    # Scale invariant SDR
    score = cal_SISNRi(src_ref, src_est, mix)

    return score


def cal_SISNRi(src_ref, src_est, mix):
    """Calculate Scale-Invariant Source-to-Noise Ratio improvement (SI-SNRi)
    Args:
        src_ref: numpy.ndarray (nsrc, nsamples) - ground truth sources
        src_est: numpy.ndarray (nsrc, nsamples) - estimated sources
        mix: numpy.ndarray, (nsamples,)
    Returns:
        average_SISNRi
    """
    sisnr1 = cal_SISNR(src_ref[0], src_est[0])
    sisnr2 = cal_SISNR(src_ref[1], src_est[1])
    sisnr1b = cal_SISNR(src_ref[0], mix)
    sisnr2b = cal_SISNR(src_ref[1], mix)
    # print("SISNR base1 {0:.2f} SISNR base2 {1:.2f}, avg {2:.2f}".format(
    #     sisnr1b, sisnr2b, (sisnr1b+sisnr2b)/2))
    # print("SISNRi1: {0:.2f}, SISNRi2: {1:.2f}".format(sisnr1, sisnr2))
    avg_SISNRi = ((sisnr1 - sisnr1b) + (sisnr2 - sisnr2b)) / 2
    
    return avg_SISNRi


def cal_SISNR(ref_sig, out_sig, eps=1e-8):
    """Calcuate Scale-Invariant Source-to-Noise Ratio (SI-SNR)
    Args:
        ref_sig: numpy.ndarray (nsamples,)
        out_sig: numpy.ndarray (nsamples,)
    Returns:
        sisnr
    """
    assert len(ref_sig) == len(out_sig)
    ref_sig = ref_sig - np.mean(ref_sig)
    out_sig = out_sig - np.mean(out_sig)
    ref_energy = np.sum(ref_sig ** 2) + eps
    proj = np.sum(ref_sig * out_sig) * ref_sig / ref_energy
    noise = out_sig - proj
    ratio = np.sum(proj ** 2) / (np.sum(noise ** 2) + eps)
    sisnr = 10 * np.log(ratio + eps) / np.log(10.0)
    
    return sisnr


#EOF
