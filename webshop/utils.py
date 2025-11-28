import numpy as np
def sample_latency_check(threshold_ms=500):
    """
    使用对数正态分布模拟网络延迟，判断是否超过阈值（默认 1 秒）
    返回 True 表示在阈值内；False 表示超出阈值
    """
    # 设置对数正态分布参数（单位：毫秒）
    mu = 0    # log(均值)；可调整，控制分布中心
    sigma = 10          # 标准差，控制离散程度

    latency_ms = np.random.lognormal(mu, sigma)
    
    return latency_ms <= threshold_ms