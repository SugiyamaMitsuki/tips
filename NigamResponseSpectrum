import numpy as np
import pandas as pd
from numba import jit, njit, prange

@jit(nopython=True)
def calc(g_acc, a11, a12, b11, b12, a21, a22, b21, b22, h, w):
    b_acc=np.zeros_like(g_acc)
    b_vel=np.zeros_like(g_acc)
    b_dis=np.zeros_like(g_acc)
    b_acc_abs=np.zeros_like(g_acc)
    b_acc[0] = -g_acc[0]
    length=len(g_acc)
    for i  in range(1,length):
        b_dis[i] = a11 * b_dis[i - 1] + a12 * b_vel[i - 1] + b11 * g_acc[i - 1] + b12 * g_acc[i]
        b_vel[i] = a21 * b_dis[i - 1] + a22 * b_vel[i - 1] + b21 * g_acc[i - 1] + b22 * g_acc[i]
        b_acc[i] = - g_acc[i] - 2.0 * h * w * b_vel[i] - w * w * b_dis[i]
        b_acc_abs[i] = b_acc[i] + g_acc[i]
    return b_dis, b_vel, b_acc_abs
    

@jit(nopython=True)
def nigam_jennings(period, g_acc, dt, h):
    """
    Calculate response using Nigam-Jennings method.

    Parameters
    ----------
    period : float
        The period for which to calculate the response.
    g_acc : np.ndarray
        The ground acceleration.
    dt : float
        The time step.
    h : float
        The damping factor.

    Returns
    -------
    tuple
        The period, maximum acceleration, maximum velocity, and maximum displacement.
    """
    if period == 0:
        _period, acc_max, vel_max, dis_max= 0.0, np.max(np.abs(g_acc)), 0, 0 
        return _period, acc_max, vel_max, dis_max
    
    # print("period",period,"dt", dt,"h", h)
    w = 2.0 * np.pi / period
    h_ = np.sqrt(1.0 - h * h)
    w_ = h_ * w
    ehw = np.exp(-h * w * dt);
    hh_ = h / h_
    sinw_ = np.sin(w_*dt)
    cosw_ = np.cos(w_ * dt)
    hw1 = (2.0 * h * h - 1.0) / (w * w * dt)
    hw2 = h / w
    hw3 = (2.0 * h) / (w * w * w * dt)
    ww = 1.0 / (w * w)
    a11 = ehw * (hh_ * sinw_ + cosw_)
    a12 = ehw / w_ * sinw_
    a21 = -w / h_ * ehw * sinw_
    a22 = ehw * (cosw_ - hh_ * sinw_)
    b11 = ehw * ((hw1 + hw2) * sinw_ / w_ + (hw3 + ww) * cosw_) - hw3
    b12 = -ehw * (hw1 * sinw_ / w_ + hw3 * cosw_) - ww + hw3
    b21 = ehw * ((hw1 + hw2) * (cosw_ - hh_ * sinw_) - (hw3 + ww) * (w_ * sinw_ + h * w * cosw_)) + ww / dt
    b22 = -ehw * (hw1 * (cosw_ - hh_ * sinw_) - hw3 * (w_ * sinw_ + h * w * cosw_)) - ww / dt
    
    b_dis, b_vel, b_acc_abs = calc(g_acc,a11,a12,b11,b12,
                                            a21,a22,b21,b22,
                                        h,w)
    dis_max, vel_max, acc_max = np.max(np.abs(b_dis)),np.max(np.abs(b_vel)),np.max(np.abs(b_acc_abs))
    
    return period, acc_max, vel_max, dis_max

# @jit(nopython=True)
# def compute_results(periods, g_acc, dt, h):
#     return [nigam_jennings(period, g_acc, dt, h) for period in periods]
@njit(parallel=True)
def compute_results(periods, g_acc, dt, h):
    n = len(periods)
    results = np.empty((n, 4))
    for i in prange(n):
        results[i, :] = nigam_jennings(periods[i], g_acc, dt, h)
    return results

#計算クラス
class Calculus:
    def __init__(self):
        None

    # 応答スペクトル計算
    def spectrum(self, g_acc, dt, periods, h):
        """
        Calculate the response spectrum.

        Parameters
        ----------
        g_acc : np.ndarray
            The ground acceleration.
        dt : float
            The time step.
        periods : list
            The periods for which to calculate the spectrum.
        h : float
            The damping factor.

        Returns
        -------
        pd.DataFrame
            The response spectrum.
        """

        print("nigam法の計算をします")
        results = compute_results(periods, g_acc, dt, h)
        df = pd.DataFrame(results, columns=['period', 'acc', 'vel', 'dis'])

        return df
    

