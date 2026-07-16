import numpy as np
from numpy.linalg import norm
# from pyexpat.model import XML_CQUANT_PLUS
from qutip.measurement import measure
from scipy.optimize import minimize
from scipy import linalg as la
from scipy.special import factorial
from qutip import *

from FusionSimulator import *
from FusionSimulator import _remove_global_phase, _Bellness, _entanglement_entropy

from QPC_Optimizer import *


# Computational Base
OO = np.array([1,0,0,0], dtype = complex)
OI = np.array([0,1,0,0], dtype = complex)
IO = np.array([0,0,1,0], dtype = complex)
II = np.array([0,0,0,1], dtype = complex)

#Bell-states
phi_plus = (OO + II) / np.sqrt(2)
phi_minus = (OO - II) / np.sqrt(2)
psi_plus = (OI + IO) / np.sqrt(2)
psi_minus = (OI - IO) / np.sqrt(2)

bell_B = np.array([phi_plus, psi_plus, phi_minus, psi_minus])
bell_B_plus = np.array([phi_plus, psi_plus])
bell_P_plus = np.outer(phi_plus, phi_plus) + np.outer(psi_plus, psi_plus)
bell_P_minus = np.outer(phi_minus, phi_minus) + np.outer(psi_minus, psi_minus)

#50/50 Beam splitter
H = np.array([[1,1],[1,-1]])/np.sqrt(2)

def tensor_prod(A,B):
    A0, A1, B0, B1 = A.shape[0],A.shape[1],B.shape[0],B.shape[1]
    return np.array([[A[i // B0][j // B1] * B[i % B0][j % B1] for j in range(A1 * B1)] for i in range(A0 * B0)], dtype=complex)

H12 = tensor_prod(H,H)

def _conjectured_scaling_gen_fusion(a):
    z = a//2
    if a%2:
        return 5/6-2**(-z)/4
    else:
        return 5/6-2**(-z)/3

#----------------------------------------------------------------------------------------------------
def computational_to_xrot_Bell_basis(state):
    #xrot on the bell basis so it fits the state (leading to state individual basis with independent rots on minus and plus)

    state_in_bell_basis = computational_to_Bell_basis(state)
    state_in_bell_basis = _remove_global_phase(state_in_bell_basis)
    x_rot_plus_v = state_in_bell_basis * np.array([ 1, 0, 1, 0], dtype=complex)
    plus_norm = np.linalg.norm(x_rot_plus_v)
    if plus_norm == 0: #mathematically not the best way to describe it, but is doesn't matter, because state is has no part in this subspace
        x_rot_plus_v = np.array([ 1, 0, 0, 0], dtype=complex)
    else:
        x_rot_plus_v = x_rot_plus_v / plus_norm
        x_rot_plus_v[2] *= np.exp(-1j * min(
            [np.pi / 2 - (np.angle(x_rot_plus_v[2])) - np.angle(x_rot_plus_v[0]),
             -np.pi / 2 - (np.angle(x_rot_plus_v[2])) - np.angle(x_rot_plus_v[0])],key = lambda x:np.abs(x)))  # only an angle-diff of \pm pi/2 is allowed

    x_rot_minus_v = state_in_bell_basis * np.array([0, 1, 0, 1], dtype=complex)
    minus_norm = np.linalg.norm(x_rot_minus_v)
    if minus_norm == 0:
        x_rot_minus_v = np.array([0, 1, 0, 0], dtype=complex)
    else:
        x_rot_minus_v = x_rot_minus_v / minus_norm
        x_rot_minus_v[3] *= np.exp(-1j * min(
            [np.pi / 2 - (np.angle(x_rot_minus_v[3])) - np.angle(x_rot_minus_v[1]),
             -np.pi / 2 - (np.angle(x_rot_minus_v[3])) - np.angle(x_rot_minus_v[1])], key=lambda x: np.abs(x)))

    other_plus_v = np.array([x_rot_plus_v[2], x_rot_plus_v[1], x_rot_plus_v[0], x_rot_plus_v[3]], dtype=complex)
    other_minus_v = np.array([x_rot_minus_v[0], x_rot_minus_v[3], x_rot_minus_v[2], x_rot_minus_v[1]], dtype=complex)

    transformation = (np.outer(Bell_to_computational_basis(x_rot_plus_v), np.array([1, 0, 0, 0], dtype=complex))
                      + np.outer(Bell_to_computational_basis(x_rot_minus_v), np.array([0, 1, 0, 0], dtype=complex))
                      + np.outer(Bell_to_computational_basis(other_plus_v), np.array([0, 0, 1, 0], dtype=complex))
                      + np.outer(Bell_to_computational_basis(other_minus_v), np.array([0, 0, 0, 1], dtype=complex)))
    return np.linalg.inv(transformation) @ state

def _is_state_bell_state_up_to_xrot(state, tolerance = 10 ** -6):
    #for comlutational speedup maybe ignore the norm for the plus and minus part (makes no big differnce)
    state_in_xrot_bell_basis = computational_to_xrot_Bell_basis(state)
    state_in_xrot_bell_basis = _remove_global_phase(state_in_xrot_bell_basis)
    nearest_xrot_plus = np.array([1,0,0,0], dtype=complex)
    nearest_xrot_minus = np.array([0, 1, 0, 0], dtype=complex)
    minimal_state_distance = min([np.linalg.norm(state_in_xrot_bell_basis - xrot_bell_state) for xrot_bell_state in
                                  (nearest_xrot_plus, nearest_xrot_minus)])
    return minimal_state_distance < tolerance
#----------------------------------------------------------------------------------------------------

#State identification

def _is_state_specific_state(state, state_measure, order, tolerance = 10 ** -6):
    measure = state_measure(state,order)

    return 1. - measure < tolerance

def _state_measure_for_X_rotated_Bell_states(state, order, tolerance = 1e-6):

    M_plus = np.abs(np.sum(phi_plus * state)**2 - np.sum(psi_plus * state)**2)**2
    M_minus = np.abs(np.sum(phi_minus * state)**2 - np.sum(psi_minus * state)**2)**2
    if order == np.inf:
        return min(1, max([M_plus, M_minus]) ** order)
    else:
        return min(1, np.sum(np.array([M_plus, M_minus]) ** order))

def _state_measure_for_XX_states(state, order):
    plus_plus = np.array([1, 1, 1, 1], dtype=complex) / 2
    minus_minus = np.array([1, -1, -1, 1], dtype=complex) / 2
    plus_minus = np.array([1, -1, 1, -1], dtype=complex) / 2
    minus_plus = np.array([1, 1, -1, -1], dtype=complex) / 2

    M_plus =  np.abs(np.sum(plus_plus * state))**2 + np.abs(np.sum(minus_minus * state))**2
    M_minus = np.abs(np.sum(plus_minus * state)) ** 2 + np.abs(np.sum(minus_plus * state)) ** 2

    if order == np.inf:
        return min(1, max([M_plus, M_minus]) ** order)
    else:
        return min(1, np.sum(np.array([M_plus, M_minus]) ** order))

def _state_measure_for_XA_rotated_ZA_and_ZB_states(state, order):

    a = np.sum(OO * state)
    b = np.sum(IO * state)
    c = np.sum(II * state)
    d = np.sum(OI * state)

    return (np.abs(a**2-b**2)**2)**order + (np.abs(c**2-d**2)**2)**order

def _state_measure_for_XB_rotated_ZA_and_ZB_states(state, order):

    a = np.sum(OO * state)
    b = np.sum(IO * state)
    c = np.sum(II * state)
    d = np.sum(OI * state)

    return (np.abs(a ** 2 - d ** 2) ** 2) ** order + (np.abs(b ** 2 - c ** 2) ** 2) ** order

def _state_measure_for_ZZ_states(state, order):
    s = np.abs(state)

    if order == np.inf:
        return min(1,max([(s[0]**2+s[3]**2)**order,(s[1]**2+s[2]**2)**order]))

    else:
        return min(1,np.sum([(s[0]**2+s[3]**2)**order,(s[1]**2+s[2]**2)**order]))

def make_pauli_rotation(angle, matrix):
    D = len(matrix)
    return np.cos(angle / 2) * np.diag(np.array([1]*D, dtype = complex)) - 1j * np.sin(angle / 2) * matrix

def _state_measure_for_XA_rotated_ZB_rotated_Bell_states(state, order):
    Z_B = np.array([[1, 0,0, 0],
                    [0,-1,0, 0],
                    [0, 0,1, 0],
                    [0, 0,0,-1]], dtype = complex)
    c_bell_plus = bell_B_plus @ state
    p_plus = np.sqrt(np.sum(np.abs(c_bell_plus)**2))
    phi2s = [np.arccos(p_plus), np.arccos(-p_plus)]
    phi2s = phi2s + list(-np.array(phi2s))
    phi2s = 2*np.array(phi2s + list(np.array(phi2s) + np.pi))

    M = max([_state_measure_for_X_rotated_Bell_states(make_pauli_rotation(-angle, Z_B)@state, order) for angle in phi2s])
    
    return min(1,M)

def _XRot_matrix(angle):
    I = np.diag([1]*4)
    X1I2 = (np.outer(np.array([0,0,1,0], dtype=complex), np.array([1,0,0,0], dtype=complex))
        + np.outer(np.array([0,0,0,1], dtype=complex), np.array([0,1,0,0], dtype=complex))
        + np.outer(np.array([1,0,0,0], dtype=complex), np.array([0,0,1,0], dtype=complex))
        + np.outer(np.array([0,1,0,0], dtype=complex), np.array([0,0,0,1], dtype=complex)))
    return np.cos(angle / 2) * I - 1j*np.sin(angle / 2) * X1I2

def _rotZZ_ness(state, order):
    #old function
    a, d  = state[0], state[3]

    angle = 2*np.arccos(min(1,np.sqrt(np.abs(a)**2 + np.abs(d)**2)))

    Rot1 = _XRot_matrix(angle)
    Rot2 = _XRot_matrix(-angle)

    state1 = Rot1 @ state
    state2 = Rot2 @ state

    return np.max([_state_measure_for_ZZ_states(state1,order),_state_measure_for_ZZ_states(state2,order)])

def _maximal_overlapping_with_XA_rotated_ZZ_state(state, tolerance = 10 ** -6):
    possible_maximal_overlapping = []

    a, b, c, d = state[0], state[1], state[2], state[3]

    a_abs, b_abs, c_abs, d_abs = np.abs([a,b,c,d])

    if (a_abs < tolerance and d_abs < tolerance) or (c_abs < tolerance and b_abs < tolerance): #is a XArotZZ state
        return 1
    


    cos_phi_half_abs = max(min(np.sqrt(a_abs**2 + d_abs**2), 1),0) 

    phi = 2*np.arccos(cos_phi_half_abs)
    sin_phi_half_abs = max(min(np.sin(phi / 2), 1), 0)

    alpha = 1
    beta = 0

    #signs of cos and sin cannot be feed into the phases of alphas and betas

    if cos_phi_half_abs > tolerance:
        alpha = a / cos_phi_half_abs
        beta = d / cos_phi_half_abs

    alpha_pp = 1
    beta_pp = 0

    if sin_phi_half_abs > tolerance:
        alpha_pp = c / (-1j * sin_phi_half_abs)
        beta_pp = b / (-1j * sin_phi_half_abs)

    # one k = 0

    if np.abs(a_abs**2 - c_abs**2) < tolerance: # assuming k_beta=0

        phi_ps = np.array([np.pi * (n + 1/2) for n in [-1,0,1,2]],dtype=complex)
        k_alphas = np.cos(phi_ps/2) * a + 1j * np.sin(phi_ps/2) * c
        k_betas = np.cos(phi_ps/2) * d + 1j * np.sin(phi_ps/2) * b
        usable = np.abs(k_betas) < tolerance # testing k_beta=0
        usable2 = np.abs(k_alphas) != 0 # testing k_alpha!=0
        usable *= usable2
        usable_k_alphas = k_alphas[usable]
        usable_overlapping = list(np.abs(usable_k_alphas)**2) # overlapping squared is the k_alpha squared
        possible_maximal_overlapping += usable_overlapping

    if np.abs(a_abs**2 - c_abs**2) != 0: # assuming k_beta=0

        phi_ps = np.array([np.arctan(np.sin(phi) * np.real(np.conjugate(alpha) * alpha_pp)/ (a_abs**2 - c_abs**2)) + np.pi * n for n in [-1,0,1,2]],dtype=complex)
        k_alphas = np.cos(phi_ps/2) * a + 1j * np.sin(phi_ps/2) * c
        k_betas = np.cos(phi_ps/2) * d + 1j * np.sin(phi_ps/2) * b
        usable = np.abs(k_betas) < tolerance # testing k_beta=0
        usable2 = np.abs(k_alphas) != 0 # testing k_alpha!=0
        usable *= usable2
        usable_k_alphas = k_alphas[usable]
        usable_overlapping = list(np.abs(usable_k_alphas)**2) # overlapping squared is the k_alpha squared
        possible_maximal_overlapping += usable_overlapping

    if np.abs(d_abs**2 - b_abs**2) < tolerance: # assuming k_alpha=0
        
        phi_ps = np.array([np.pi * (n + 1/2) for n in [-1,0,1,2]],dtype=complex)
        k_alphas = np.cos(phi_ps/2) * a + 1j * np.sin(phi_ps/2) * c
        k_betas = np.cos(phi_ps/2) * d + 1j * np.sin(phi_ps/2) * b
        usable = np.abs(k_alphas) < tolerance # testing k_alpha=0
        usable2 = np.abs(k_betas) != 0 # testing k_beta!=0
        usable *= usable2
        usable_k_betas = k_betas[usable]
        usable_overlapping = list(np.abs(usable_k_betas)**2) # overlapping squared is the k_beta squared
        possible_maximal_overlapping += usable_overlapping

    if np.abs(d_abs**2 - b_abs**2) != 0: # assuming k_alpha=0

        phi_ps = np.array([np.arctan(np.sin(phi) * np.real(np.conjugate(beta) * beta_pp)/ (b_abs**2 - d_abs**2)) + np.pi * n for n in [-1,0,1,2]],dtype=complex)
        k_alphas = np.cos(phi_ps/2) * a + 1j * np.sin(phi_ps/2) * c
        k_betas = np.cos(phi_ps/2) * d + 1j * np.sin(phi_ps/2) * b
        usable = np.abs(k_alphas) < tolerance # testing k_alpha=0
        usable2 = np.abs(k_betas) != 0 # testing k_beta!=0
        usable *= usable2
        usable_k_betas = k_betas[usable]
        usable_overlapping = list(np.abs(usable_k_betas)**2) # overlapping squared is the k_beta squared
        possible_maximal_overlapping += usable_overlapping

    # no k = 0  
    if np.cos(phi) != 0: # assuming k_alpha!=0 and k_beta!=0

        phi_ps = np.array([np.arctan(np.tan(phi) * np.real(np.conjugate(alpha) * alpha_pp + np.conjugate(beta) * beta_pp)) + n * np.pi for n in [-1,0,1,2]])
        k_alpha_abss = np.abs(np.cos(phi_ps/2) * a + 1j * np.sin(phi_ps/2) * c)
        k_beta_abss = np.abs(np.cos(phi_ps/2) * d + 1j * np.sin(phi_ps/2) * b)
        usable = k_alpha_abss != 0 #tolerance # testing k_alpha!=0
        usable2 = k_beta_abss != 0 #tolerance # testing k_beta!=0
        usable *= usable2
        usable_k_alpha_abss = k_alpha_abss[usable]
        usable_k_beta_abss = k_beta_abss[usable]
        usable_alpha_p_abss = 1/np.sqrt(1 + usable_k_beta_abss**2/usable_k_alpha_abss**2)
        usable_beta_p_abss = 1/np.sqrt(1 + usable_k_alpha_abss**2/usable_k_beta_abss**2)
        usable_overlapping = list((usable_alpha_p_abss * usable_k_alpha_abss + usable_beta_p_abss * usable_k_beta_abss)**2)
        possible_maximal_overlapping += usable_overlapping
    
    if np.cos(phi) < tolerance: # assuming k_alpha!=0 and k_beta!=0

        phi_ps = np.array([np.pi * (n + 1/2) for n in [-1,0,1,2]],dtype=complex)
        k_alpha_abss = np.abs(np.cos(phi_ps/2) * a + 1j * np.sin(phi_ps/2) * c)
        k_beta_abss = np.abs(np.cos(phi_ps/2) * d + 1j * np.sin(phi_ps/2) * b)
        usable = k_alpha_abss != 0 # testing k_alpha!=0
        usable2 = k_beta_abss != 0 # testing k_beta!=0
        usable *= usable2
        usable_k_alpha_abss = k_alpha_abss[usable]
        usable_k_beta_abss = k_beta_abss[usable]
        usable_alpha_p_abss = 1/np.sqrt(1 + usable_k_beta_abss**2/usable_k_alpha_abss**2)
        usable_beta_p_abss = 1/np.sqrt(1 + usable_k_alpha_abss**2/usable_k_beta_abss**2)
        usable_overlapping = list((usable_alpha_p_abss * usable_k_alpha_abss + usable_beta_p_abss * usable_k_beta_abss)**2)
        possible_maximal_overlapping += usable_overlapping

    if len(possible_maximal_overlapping) == 0:
        return 0
    
    else:
        return min(np.max(possible_maximal_overlapping),1)

def _maximal_overlapping_with_XB_rotated_ZZ_state(state, tolerance = 10 ** -6):
    possible_maximal_overlapping = []

    a, c, b, d = state[0], state[1], state[2], state[3]

    a_abs, b_abs, c_abs, d_abs = np.abs([a,b,c,d])

    if (a_abs < tolerance and d_abs < tolerance) or (c_abs < tolerance and b_abs < tolerance): #is a rotZZ state
        return 1
    
    

    cos_phi_half_abs = max(min(np.sqrt(a_abs**2 + d_abs**2), 1),0) 

    phi = 2*np.arccos(cos_phi_half_abs)
    sin_phi_half_abs = max(min(np.sin(phi / 2), 1), 0)

    alpha = 1
    beta = 0

    #signs of cos and sin cannot be feed into the phases of alphas and betas

    if cos_phi_half_abs > tolerance:
        alpha = a / cos_phi_half_abs
        beta = d / cos_phi_half_abs

    alpha_pp = 1
    beta_pp = 0

    if sin_phi_half_abs > tolerance:
        alpha_pp = c / (-1j * sin_phi_half_abs)
        beta_pp = b / (-1j * sin_phi_half_abs)

    # one k = 0

    if np.abs(a_abs**2 - c_abs**2) < tolerance: # assuming k_beta=0

        phi_ps = np.array([np.pi * (n + 1/2) for n in [-1,0,1,2]],dtype=complex)
        k_alphas = np.cos(phi_ps/2) * a + 1j * np.sin(phi_ps/2) * c
        k_betas = np.cos(phi_ps/2) * d + 1j * np.sin(phi_ps/2) * b
        usable = np.abs(k_betas) < tolerance # testing k_beta=0
        usable2 = np.abs(k_alphas) != 0 # testing k_alpha!=0
        usable *= usable2
        usable_k_alphas = k_alphas[usable]
        usable_overlapping = list(np.abs(usable_k_alphas)**2) # overlapping squared is the k_alpha squared
        possible_maximal_overlapping += usable_overlapping

    if np.abs(a_abs**2 - c_abs**2) != 0: # assuming k_beta=0

        phi_ps = np.array([np.arctan(np.sin(phi) * np.real(np.conjugate(alpha) * alpha_pp)/ (a_abs**2 - c_abs**2)) + np.pi * n for n in [-1,0,1,2]],dtype=complex)
        k_alphas = np.cos(phi_ps/2) * a + 1j * np.sin(phi_ps/2) * c
        k_betas = np.cos(phi_ps/2) * d + 1j * np.sin(phi_ps/2) * b
        usable = np.abs(k_betas) < tolerance # testing k_beta=0
        usable2 = np.abs(k_alphas) != 0 # testing k_alpha!=0
        usable *= usable2
        usable_k_alphas = k_alphas[usable]
        usable_overlapping = list(np.abs(usable_k_alphas)**2) # overlapping squared is the k_alpha squared
        possible_maximal_overlapping += usable_overlapping

    if np.abs(d_abs**2 - b_abs**2) < tolerance: # assuming k_alpha=0
        
        phi_ps = np.array([np.pi * (n + 1/2) for n in [-1,0,1,2]],dtype=complex)
        k_alphas = np.cos(phi_ps/2) * a + 1j * np.sin(phi_ps/2) * c
        k_betas = np.cos(phi_ps/2) * d + 1j * np.sin(phi_ps/2) * b
        usable = np.abs(k_alphas) < tolerance # testing k_alpha=0
        usable2 = np.abs(k_betas) != 0 # testing k_beta!=0
        usable *= usable2
        usable_k_betas = k_betas[usable]
        usable_overlapping = list(np.abs(usable_k_betas)**2) # overlapping squared is the k_beta squared
        possible_maximal_overlapping += usable_overlapping

    if np.abs(d_abs**2 - b_abs**2) != 0: # assuming k_alpha=0

        phi_ps = np.array([np.arctan(np.sin(phi) * np.real(np.conjugate(beta) * beta_pp)/ (b_abs**2 - d_abs**2)) + np.pi * n for n in [-1,0,1,2]],dtype=complex)
        k_alphas = np.cos(phi_ps/2) * a + 1j * np.sin(phi_ps/2) * c
        k_betas = np.cos(phi_ps/2) * d + 1j * np.sin(phi_ps/2) * b
        usable = np.abs(k_alphas) < tolerance # testing k_alpha=0
        usable2 = np.abs(k_betas) != 0 # testing k_beta!=0
        usable *= usable2
        usable_k_betas = k_betas[usable]
        usable_overlapping = list(np.abs(usable_k_betas)**2) # overlapping squared is the k_beta squared
        possible_maximal_overlapping += usable_overlapping

    # no k = 0  
    if np.cos(phi) != 0: # assuming k_alpha!=0 and k_beta!=0

        phi_ps = np.array([np.arctan(np.tan(phi) * np.real(np.conjugate(alpha) * alpha_pp + np.conjugate(beta) * beta_pp)) + n * np.pi for n in [-1,0,1,2]])
        k_alpha_abss = np.abs(np.cos(phi_ps/2) * a + 1j * np.sin(phi_ps/2) * c)
        k_beta_abss = np.abs(np.cos(phi_ps/2) * d + 1j * np.sin(phi_ps/2) * b)
        usable = k_alpha_abss != 0 #tolerance # testing k_alpha!=0
        usable2 = k_beta_abss != 0 #tolerance # testing k_beta!=0
        usable *= usable2
        usable_k_alpha_abss = k_alpha_abss[usable]
        usable_k_beta_abss = k_beta_abss[usable]
        usable_alpha_p_abss = 1/np.sqrt(1 + usable_k_beta_abss**2/usable_k_alpha_abss**2)
        usable_beta_p_abss = 1/np.sqrt(1 + usable_k_alpha_abss**2/usable_k_beta_abss**2)
        usable_overlapping = list((usable_alpha_p_abss * usable_k_alpha_abss + usable_beta_p_abss * usable_k_beta_abss)**2)
        possible_maximal_overlapping += usable_overlapping
    
    if np.cos(phi) < tolerance: # assuming k_alpha!=0 and k_beta!=0

        phi_ps = np.array([np.pi * (n + 1/2) for n in [-1,0,1,2]],dtype=complex)
        k_alpha_abss = np.abs(np.cos(phi_ps/2) * a + 1j * np.sin(phi_ps/2) * c)
        k_beta_abss = np.abs(np.cos(phi_ps/2) * d + 1j * np.sin(phi_ps/2) * b)
        usable = k_alpha_abss != 0 # testing k_alpha!=0
        usable2 = k_beta_abss != 0 # testing k_beta!=0
        usable *= usable2
        usable_k_alpha_abss = k_alpha_abss[usable]
        usable_k_beta_abss = k_beta_abss[usable]
        usable_alpha_p_abss = 1/np.sqrt(1 + usable_k_beta_abss**2/usable_k_alpha_abss**2)
        usable_beta_p_abss = 1/np.sqrt(1 + usable_k_alpha_abss**2/usable_k_beta_abss**2)
        usable_overlapping = list((usable_alpha_p_abss * usable_k_alpha_abss + usable_beta_p_abss * usable_k_beta_abss)**2)
        possible_maximal_overlapping += usable_overlapping

    if len(possible_maximal_overlapping) == 0:
        return 0
    
    else:
        return min(np.max(possible_maximal_overlapping),1)

def _maximal_overlapping_with_XA_rotated_XB_rotated_ZZ_state(state, tolerance = 10 ** -6):

    plus_plus = np.array([1, 1, 1, 1], dtype=complex) / 2
    minus_minus = np.array([1, -1, -1, 1], dtype=complex) / 2
    plus_minus = np.array([1, -1, 1, -1], dtype=complex) / 2
    minus_plus = np.array([1, 1, -1, -1], dtype=complex) / 2
    
    ress = []

    a_abs = np.abs(np.sum(plus_plus * state))
    d_abs = np.abs(np.sum(minus_minus * state))
    b_abs = np.abs(np.sum(plus_minus * state))
    c_abs = np.abs(np.sum(minus_plus * state))
    
    apd = a_abs + d_abs
    bpc = b_abs + c_abs

    if apd < tolerance:
        ress.append(1/2 * bpc**2)

    if bpc < tolerance:
        ress.append(1/2 * apd**2)
    
    if apd != 0 and bpc != 0:
        gamma_p = np.sqrt(1/2 * 1/(1 + (bpc/apd)**2))
        gamma_m = np.sqrt(1/2 * 1/(1 + (apd/bpc)**2))
        ress.append((gamma_p * apd + gamma_m * bpc)**2)

    return max(ress)

def state_measure_type_to_state_measure(state_measure_type):
 
    if state_measure_type == 0:
         return _state_measure_for_X_rotated_Bell_states

    elif state_measure_type == 1:
        return _state_measure_for_XX_states

    elif state_measure_type == 2:
        return _state_measure_for_XA_rotated_ZA_and_ZB_states

    elif state_measure_type == 3:
        return _state_measure_for_XB_rotated_ZA_and_ZB_states

    elif state_measure_type == 4:
        return _rotZZ_ness

    elif state_measure_type == 5:
        return _state_measure_for_XA_rotated_ZB_rotated_Bell_states

    elif state_measure_type == 6:
        return _Bellness

    elif state_measure_type == 7:
        return lambda x,y: _entanglement_entropy(x)**y

    elif state_measure_type == 8:
        return lambda x,y: _state_measure_for_X_rotated_Bell_states(H12@x,y)

# FusionSimulator addon

class MatrixQPCFusion_with_arbitrary_state_measure(MatrixQPCFusion):

    def choose_state_measure(self,state_measure_type = 0):
        self.state_measure = state_measure_type_to_state_measure(state_measure_type)


    def Bellness_of_all_states(self, order=4, weight = 0):
        result = 0
        for id in self.measurement_outcome:
            result += self.measurement_outcome[id][1] * ((1-weight)*_Bellness(self.measurement_outcome[id][0], order)
                                                         + weight*_state_measure_for_ZZ_states(self.measurement_outcome[id][0],order))
        return result

    def xrot_Bellness_of_all_states(self, order=2, weight = 0):
        result = 0

        if weight != 0:
            for id in self.measurement_outcome:
                result += self.measurement_outcome[id][1] * ((1 - weight) * _state_measure_for_X_rotated_Bell_states(self.measurement_outcome[id][0], order)
                                                                + (weight) * _maximal_overlapping_with_XA_rotated_ZZ_state(self.measurement_outcome[id][0])**(2 * order))
                                                                # + (weight)*_rotZZ_ness(self.measurement_outcome[id][0], 2*order))
        else:

            for id in self.measurement_outcome:
                result += self.measurement_outcome[id][1] * _state_measure_for_X_rotated_Bell_states(self.measurement_outcome[id][0], order)

        return result

    def probability_weighted_state_measure_of_all_state(self, order = 4):
        result = 0
        for id in self.measurement_outcome:
            result += self.measurement_outcome[id][1] * self.state_measure(self.measurement_outcome[id][0], order)
        return result

    def select_bell_measurement_up_to_xrot_pattern(self, tolerance = 10**-6):
        self.successful_patterns = {}
        for i in self.measurement_outcome:
            if _is_state_bell_state_up_to_xrot(self.measurement_outcome[i][0], tolerance):
                self.successful_patterns[i] = (self.measurement_outcome[i])

    def select_successful_measurement_patterns(self, order = 4,tolerance = 10**-6):
        self.successful_patterns = {}
        for i in self.measurement_outcome:
            if _is_state_specific_state(self.measurement_outcome[i][0], self.state_measure, order, tolerance):
                self.successful_patterns[i] = (self.measurement_outcome[i])

class MatrixQPCFusion_one_side(MatrixQPCFusion_with_arbitrary_state_measure):
    
    def __init__(self, qpc_parameter, ancilla_photon_distribution, precomputed_creation_operator=None):
        super().__init__(qpc_parameter, ancilla_photon_distribution, precomputed_creation_operator)
        
        # always at the start
        self.bell_unitary = np.diag([1+0j]*self.total_number_of_modes)
        v = np.sqrt(2,dtype=complex)
        self.bell_unitary[0][0] = 1/v
        self.bell_unitary[1][1] = 1/v
        self.bell_unitary[2][2] = -1/v
        self.bell_unitary[3][3] = -1/v
        self.bell_unitary[0][3] = 1/v
        self.bell_unitary[1][2] = 1/v
        self.bell_unitary[2][1] = 1/v
        self.bell_unitary[3][0] = 1/v

    def apply_unitary_from_clements(self, coefficients, order=None):

        unitary = unitary_from_clements(self.total_number_of_modes - 2, coefficients, order=order)
        upper_side = np.eye(self.total_number_of_modes)[:2].T[2:].T
        unitary = np.concatenate((upper_side, unitary), axis = 0)
        right_side = np.eye(self.total_number_of_modes).T[:2].T
        unitary = np.concatenate((right_side, unitary), axis = 1)
        self.apply_unitary(unitary@self.bell_unitary)

def add_init_to_split(unitary):
    total_number_of_modes = unitary.shape[0] + 2
    bell_unitary = np.diag([1+0j]*total_number_of_modes)
    v = np.sqrt(2,dtype=complex)
    bell_unitary[0][0] = 1/v + 0j
    bell_unitary[1][1] = 1/v + 0j
    bell_unitary[2][2] = -1/v + 0j
    bell_unitary[3][3] = -1/v + 0j
    bell_unitary[0][3] = 1/v + 0j
    bell_unitary[1][2] = 1/v + 0j
    bell_unitary[2][1] = 1/v + 0j
    bell_unitary[3][0] = 1/v + 0j

    upper_side = np.eye(total_number_of_modes)[:2].T[2:].T
    u = np.concatenate((upper_side, unitary), axis = 0)
    right_side = np.eye(total_number_of_modes).T[:2].T
    u = np.concatenate((right_side, u), axis = 1)
    
    u = u@bell_unitary

    return u


# QPC_Optimizer addon

def xrot_Bellness_of_all_states(clements_coeffs, arrangement_order, polynom_order, precomputed,
                               qpc_parameter=(1,1), ancilla =(), weight = 0):
    psi = MatrixQPCFusion_with_arbitrary_state_measure(qpc_parameter, ancilla, precomputed_creation_operator= precomputed)
    psi.apply_unitary_from_clements(clements_coeffs, order= arrangement_order)
    psi.measure()
    return psi.xrot_Bellness_of_all_states(order=polynom_order, weight = weight)

def Bellness_of_all_states(clements_coeffs, arrangement_order, polynom_order, precomputed,
                               qpc_parameter=(2,1), ancilla =(), weight = 0):
    psi = MatrixQPCFusion_with_arbitrary_state_measure(qpc_parameter, ancilla, precomputed_creation_operator=precomputed)
    psi.apply_unitary_from_clements(clements_coeffs, order= arrangement_order)
    psi.measure()
    return psi.Bellness_of_all_states(order=polynom_order,weight = weight)

def probability_weighted_state_measure_of_all_states(clements_coeffs, arrangement_order, polynom_order, precomputed,
                               qpc_parameter=(1,1), ancilla =(), state_measure_type = 0, one_side = False):
    
    state_measure = state_measure_type_to_state_measure(state_measure_type)

    if not one_side: 
        psi = MatrixQPCFusion_with_arbitrary_state_measure(qpc_parameter, ancilla, precomputed_creation_operator = precomputed)
        psi.state_measure = state_measure
        psi.apply_unitary_from_clements(clements_coeffs, order= arrangement_order)
        psi.measure()
        return psi.probability_weighted_state_measure_of_all_state(order=polynom_order)
    
    else: 
        psi = MatrixQPCFusion_one_side(qpc_parameter, ancilla, precomputed_creation_operator = precomputed)
        psi.state_measure = state_measure
        psi.apply_unitary_from_clements(clements_coeffs, order= arrangement_order)
        psi.measure()
        return psi.probability_weighted_state_measure_of_all_state(order=polynom_order)

def QPC_optimizer_probability_weighted_state_measure_of_all_states(clements_coeffs, arrangement_order, precomputed, qpc_parameter = (1,1), ancilla =(), polynom_order=1, state_measure_type = 0, split=False):
    
    target_function = lambda coeffs: (-1) * probability_weighted_state_measure_of_all_states(coeffs, arrangement_order, polynom_order, precomputed, qpc_parameter = qpc_parameter, ancilla = ancilla, state_measure_type = state_measure_type, one_side = split)
    optimizer = minimize(target_function, clements_coeffs)

    return (optimizer.x, -optimizer.fun)

def setting_up_Simulator_with_unitary_xrot(unitary,qpc_parameter=(1,1),ancilla=[]):
    psi = MatrixQPCFusion_with_arbitrary_state_measure(qpc_parameter,ancilla)
    psi.apply_unitary(unitary)
    psi.measure()
    return psi

# other stuff

def print_unitary_for_Mathematica(unitary):
    unitary_string = '{'

    for row in unitary:
        unitary_string += '{'

        for element in row:
            unitary_string += f'{element},'
        unitary_string = unitary_string[:-1] + '},'
    unitary_string = unitary_string[:-1] + '}'
    print(unitary_string)

def unitary_to_higher_mode_number_unitary(U,additional_modes):
    N = 0
    if type(U) == list:
        N = len(U) + additional_modes
    else:
        N = U.shape[0] + additional_modes
    higher_order_U = np.zeros((N,N),dtype=complex)

    for i in range(N):
        for j in range(N):
            if i < N - additional_modes and j < N - additional_modes:
                higher_order_U[i][j] = U[i][j]
            elif i==j:
                higher_order_U[i][j] = 1
    return higher_order_U

def save_file_for_plotting(fname_input,fname,number_of_modes,print_bound = 1):
    data = read_all_data_from_file(fname_input)
    probs = []
    coeffs = []
    for i in range(int(len(data) / 2)):
        coeffs.append(data[2 * i])
        probs.append(data[2 * i + 1])
    index_of_best_run = np.argmax(np.array(probs))
    best_coeffs = coeffs[index_of_best_run]
    best_unitary = unitary_from_clements(number_of_modes,best_coeffs)
    open(fname, 'w').close()
    with open(fname, 'a') as f:
        for i in range(len(probs)):
            dis = unitary_distance(best_unitary,unitary_from_clements(number_of_modes,coeffs[i]),number_of_modes)
            f.write(f'{dis} {probs[i]}\n')
            if probs[i] > print_bound:
                print(coeffs[i],number_of_modes,probs[i])

def unitary_distance(U1,U2,dim):
    return np.sqrt(dim-np.abs(np.trace(np.transpose(np.conjugate(U2)) @ U1)))

def save_coeff_order(fname,order):
    open(fname, 'w').close()
    with open(fname, 'a') as f:
        for i in range(len(order)):
            a = f'{order[i]}'[1:-1] + '\n'
            f.write(a)

def load_coeff_order(fname):
    order = []
    with open(fname, 'r') as f:
        for line in f:
            line = line[:-1].split(', ')
            order.append((int(line[0]),int(line[1])))
    return order
