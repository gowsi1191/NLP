# curves.py
import numpy as np
import math

def sigmoid_e(x, k=8, midpoint=0.5):
    return 1 / (1 + np.exp(-k * (x - midpoint)))

# def custom_e_curve(x, k=8, midpoint=0.5):
#     if x < 0.85:
#         return 1 / (1 + math.exp(-k * (x - midpoint)))  # Inverted sigmoid
#     else:
#         return x  # Linear increase
    
#     import math

# def custom_e_curve(x, k=8, sigmoid_mid=0.15, inverse_mid=0.5):
#     """
#     Piecewise E-score transformation:
#     - Sigmoid (0 ≤ x < 0.3): Smooth rise for very low E.
#     - Inverted Sigmoid (0.3 ≤ x < 0.8): Penalize mid-range E.
#     - Linear (x ≥ 0.8): Trust high E directly.
#     Output is guaranteed in [0, 1].
#     """
#     if x < 0.2:
#         # Standard sigmoid (low E -> gradual increase)
#         return 1 / (1 + math.exp(-k * (x - sigmoid_mid)))
#     elif 0.5 <= x < 0.6:
#         # Inverted sigmoid (mid E -> penalize false positives)
#         return 1 - (1 / (1 + math.exp(-k * (x - inverse_mid))))
#     else:
#         # Linear (high E -> trust directly, clamped to [0,1])
#         return min(max(x, 0.0), 1.0)
    
def custom_e_curve(x, k=8, sigmoid_mid=0.15, inverse_mid=0.5, 
                   low_thresh=0.2, mid_thresh=0.5, high_thresh=0.8):
    """
    Fully dynamic piecewise E-score transformation:
    - Sigmoid (x < low_thresh)
    - Linear blend (low_thresh ≤ x < mid_thresh)
    - Inverted Sigmoid (mid_thresh ≤ x < high_thresh)
    - Linear (x ≥ high_thresh)
    """
    if x < low_thresh:
        return 1 / (1 + math.exp(-k * (x - sigmoid_mid)))
    
    elif x < mid_thresh:
        # Interpolation between sigmoid and inverted sigmoid
        sig_val = 1 / (1 + math.exp(-k * (low_thresh - sigmoid_mid)))
        inv_val = 1 - (1 / (1 + math.exp(-k * (mid_thresh - inverse_mid))))
        alpha = (x - low_thresh) / (mid_thresh - low_thresh)
        return sig_val + alpha * (inv_val - sig_val)
    
    elif x < high_thresh:
        inv_val = 1 - (1 / (1 + math.exp(-k * (x - inverse_mid))))
        linear_val = high_thresh  # Because we interpolate toward x here
        alpha = (x - mid_thresh) / (high_thresh - mid_thresh)
        return inv_val + alpha * (linear_val - inv_val)
    
    else:
        return min(max(x, 0.0), 1.0)

    
# import math

# def custom_e_curve(x, k=8, sigmoid_mid=0.15, inverse_mid=0.5, 
#                    sigmoid_thresh=0.2, inverse_thresh=0.75):
#     """
#     Dynamically controlled Piecewise E-score transformation:
#     - 0 ≤ x < sigmoid_thresh → standard sigmoid rise
#     - sigmoid_thresh ≤ x < inverse_thresh → inverted sigmoid drop
#     - x ≥ inverse_thresh → linear trust

#     Args:
#         k: steepness
#         sigmoid_mid: center of low-end sigmoid
#         inverse_mid: center of mid-range inverted sigmoid
#         sigmoid_thresh: upper bound for sigmoid section
#         inverse_thresh: lower bound for linear section
#     """
#     if x < sigmoid_thresh:
#         return 1 / (1 + math.exp(-k * (x - sigmoid_mid)))
#     elif x < inverse_thresh:
#         return 1 - (1 / (1 + math.exp(-k * (x - inverse_mid))))
#     else:
#         return x  # no clamping as per your request
 
    
# import math

# def custom_e_curve(x, k=10, sigmoid_mid=0.1, inverse_mid=0.45):
#     """
#     Data-informed E-score transformation:
#     - 0 ≤ x < 0.2: Use sigmoid (low E likely irrelevant)
#     - 0.2 ≤ x < 0.75: Inverted sigmoid to penalize ambiguous mid-E
#     - ≥ 0.75: Trust as relevant (linear)
#     """
#     if x < 0.2:
#         return 1 / (1 + math.exp(-k * (x - sigmoid_mid)))
#     elif x < 0.75:
#         return 1 - (1 / (1 + math.exp(-k * (x - inverse_mid))))
#     else:
#         return x  # allow >1 if x > 1 due to scoring



def smooth_join_score(x, join_x=0.6, join_y=0.4, exp1=0.6, exp2=0.5):
    if x < join_x:
        scale = join_y / (join_x ** exp1)
        return scale * (x ** exp1)
    else:
        norm_x = (x - join_x) / (1 - join_x)
        return join_y + (1 - join_y) * (norm_x ** exp2)

def power_root_curve(x, exponent=0.6):
    return x ** exponent

def inverted_sigmoid(x, k=8, midpoint=0.5):
    return 1 - sigmoid_e(x, k, midpoint)

def arctangent_scaled(x, k=10):
    return (np.arctan(k * (x - 0.5)) / np.arctan(k * 0.5) + 1) / 2

def smoothstep(x):
    return 3 * x**2 - 2 * x**3

def smootherstep(x):
    return 6 * x**5 - 15 * x**4 + 10 * x**3

def linear(x):
    return x

