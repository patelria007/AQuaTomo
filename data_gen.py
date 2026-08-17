import os

# Force PyTorch MPS to use CPU for missing complex linear algebra kernels
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import platform
import numpy as np # Used ONLY for classical pre-processing and qiskit bridging
import pandas as pd
from itertools import product
from scipy.stats import norm
from qiskit.quantum_info import random_density_matrix, random_statevector
import argparse

# 1. HARDWARE DETECTION (THE DRIVER)
def get_compute_backend(use_gpu=True):
    """
    Dynamically detects available hardware and returns the Array API namespace (xp) 
    and the target device. This prevents ModuleNotFoundErrors across environments.
    """
    import array_api_compat.numpy as xp_np
    
    if not use_gpu:
        print("Backend: NumPy (CPU - Forced)")
        return xp_np, "cpu"
    
    # Check for Apple Silicon (Macs)
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        try:
            import torch
            import array_api_compat.torch as xp_torch
            print("Backend: PyTorch (Apple MPS)")
            return xp_torch, torch.device("mps")
        except ImportError:
            pass
            
    # Check for NVIDIA GPUs (Cluster)
    try:
        import cupy
        import array_api_compat.cupy as xp_cupy
        print("Backend: CuPy (NVIDIA CUDA)")
        return xp_cupy, cupy.cuda.Device(0)
    except ImportError:
        pass
        
    print("Backend: NumPy (CPU - Fallback)")
    return xp_np, "cpu"

# 2. CLASSICAL PRE-PROCESSING (BASIS GENERATION)
PAULI = {
    'I': np.eye(2, dtype=complex),
    'X': np.array([[0,1],[1,0]], dtype=complex),
    'Y': np.array([[0,-1j],[1j,0]], dtype=complex),
    'Z': np.array([[1,0],[0,-1]], dtype=complex),
}

def pauli_group(n):
    labels = [''.join(p) for p in product('IXYZ', repeat=n)]
    ops = []
    for lab in labels:
        M = np.array([[1]], dtype=complex)
        for ch in lab:
            M = np.kron(M, PAULI[ch])
        ops.append(M)
    return np.array(ops)

# 3. HARDWARE-AGNOSTIC COMPUTE ENGINE (ARRAY API)
def global_depo_channel(dm, p, xp):
    dim = dm.shape[0]
    # xp.eye correctly allocates an identity matrix on the target hardware
    I = xp.eye(dim, dtype=dm.dtype, device=dm.device)
    return (1 - p) * dm + (p / dim) * I

def extract_cholesky_components(T_matrix, xp):
    """Helper function to extract the real/imag components from a lower-triangular matrix."""
    dim = T_matrix.shape[0]
    diag_idx = np.arange(dim)
    row_idx, col_idx = np.tril_indices(dim, k=-1)
    
    diag = xp.real(T_matrix[diag_idx, diag_idx])
    reals = xp.real(T_matrix[row_idx, col_idx])
    imags = xp.imag(T_matrix[row_idx, col_idx])
    
    return xp.concat([diag, reals, imags])

def vectorization_new(rho, xp):
    # dim = rho.shape[0]
    # chol = xp.linalg.cholesky(rho)
    
    # # Calculate extraction indices on CPU (small metadata overhead)
    # diag_idx = np.arange(dim)
    # row_idx, col_idx = np.tril_indices(dim, k=-1)
    
    # # Extract directly on the device
    # diag = xp.real(chol[diag_idx, diag_idx])
    # reals = xp.real(chol[row_idx, col_idx])
    # imags = xp.imag(chol[row_idx, col_idx])
    
    # return xp.concat([diag, reals, imags])

    dim = rho.shape[0]
    
    # 1. Force strict Hermitian structure to eliminate floating-point asymmetries
    rho = (rho + xp.conj(rho).T) / 2.0
    
    # 2. Add numerical jitter to the diagonal to guarantee positive-definiteness
    jitter = 1e-5
    I = xp.eye(dim, dtype=rho.dtype, device=rho.device)
    rho_stabilized = rho + jitter * I
    
    chol = xp.linalg.cholesky(rho_stabilized)
    
    return extract_cholesky_components(chol, xp)

def eigenvaluesCheck(dm, xp):
    # eigh is strictly for Hermitian matrices, standard across all Array APIs
    eis, eigvecs = xp.linalg.eigh(dm) 
    
    # Threshold negative eigenvalues to 0.0001 dynamically on device
    eis_clipped = xp.where(eis < 0.0, xp.asarray(0.0001, dtype=eis.dtype, device=eis.device), eis)
    
    # Reconstruct: V @ W @ V.dagger
    cleanRho = eigvecs @ (xp.expand_dims(eis_clipped, axis=1) * xp.conj(eigvecs).T)
    return cleanRho / xp.real(xp.trace(cleanRho))

def PureEigenvaluesCheck(dm, xp):
    eis, eigvecs = xp.linalg.eigh(dm)
    eis_clipped = xp.where(eis < 0.99, xp.asarray(0.0001, dtype=eis.dtype, device=eis.device), eis)
    cleanRho = eigvecs @ (xp.expand_dims(eis_clipped, axis=1) * xp.conj(eigvecs).T)
    return cleanRho / xp.real(xp.trace(cleanRho))

def get_theoretical_cholesky(rho_xp, xp):
    """
    Computes the ideal Cholesky target using 64-bit CPU math.
    This safely bypasses all 32-bit hardware factorization bugs.
    """
    # 1. Bring matrix to CPU
    if hasattr(rho_xp, 'cpu'):
        rho_np = rho_xp.cpu().numpy()
    elif hasattr(rho_xp, 'get'):
        rho_np = rho_xp.get()
    else:
        rho_np = np.array(rho_xp)
        
    # 2. Force high-precision 64-bit complex
    rho_np = np.array(rho_np, dtype=np.complex128)
    
    # 3. Stabilize mathematically and decompose
    rho_np = (rho_np + rho_np.conj().T) / 2.0
    rho_np += 1e-12 * np.eye(rho_np.shape[0])
    chol_np = np.linalg.cholesky(rho_np)
    
    # 4. Extract components on CPU
    dim = chol_np.shape[0]
    diag_idx = np.arange(dim)
    row_idx, col_idx = np.tril_indices(dim, k=-1)
    
    diag = np.real(chol_np[diag_idx, diag_idx])
    reals = np.real(chol_np[row_idx, col_idx])
    imags = np.imag(chol_np[row_idx, col_idx])
    
    components_np = np.concatenate([diag, reals, imags])
    
    # 5. Return to original device and match precision
    f_dtype = xp.float32 if rho_xp.dtype == xp.complex64 else xp.float64
    return xp.asarray(components_np, dtype=f_dtype, device=rho_xp.device)

def xp_sqrtm(m, xp):
    """Helper to calculate matrix square root on hardware using eigendecomposition"""
    w, v = xp.linalg.eigh(m)
    w_sqrt = xp.sqrt(xp.where(w < 0.0, xp.asarray(0.0, dtype=w.dtype, device=w.device), w))
    return v @ (xp.expand_dims(w_sqrt, axis=1) * xp.conj(v).T)

def xp_fidelity(rho1, rho2, xp):
    """Hardware-agnostic fidelity calculation (replaces QuTiP's CPU-bound fidelity)"""
    sq_rho1 = xp_sqrtm(rho1, xp)
    inner = sq_rho1 @ rho2 @ sq_rho1
    sqrt_inner = xp_sqrtm(inner, xp)
    fid = xp.real(xp.trace(sqrt_inner))**2
    return fid

def HSdist(A, B, xp):
    diff = A - B
    return xp.real(xp.trace(diff @ xp.conj(diff).T))

# 4. MAIN GENERATION PIPELINE
def mle_reconstruction(borns_approx, basis, xp, lr=0.01, iters=300):
    """
    Hardware-agnostic Maximum Likelihood Estimation using Gradient Descent.
    Parameterizes the density matrix as rho = (T @ T.dagger) / Tr(T @ T.dagger)
    to strictly enforce positive semi-definiteness and unit trace.
    """
    dim = basis.shape[1]
    num_ops = basis.shape[0]
    
    # 1. Initialize T as a scaled identity matrix
    T = xp.eye(dim, dtype=basis.dtype, device=basis.device)

    # Derive the correct real float type directly from the input array
    f_dtype = xp.float32 if basis.dtype == xp.complex64 else xp.float64
    
    # Pre-allocate limits for Array API stability
    min_trace = xp.asarray(1e-6, dtype=f_dtype, device=basis.device)
    clip_limit = xp.asarray(1.0, dtype=f_dtype, device=basis.device)

    for _ in range(iters):
        # 2. Forward Pass: Construct physical rho
        A = T @ xp.conj(T).T
        tr_A = xp.real(xp.trace(A))
        
        # Clamp trace to prevent division by zero / exploding gradients
        tr_A_safe = xp.where(tr_A < min_trace, min_trace, tr_A)
        rho = A / tr_A_safe
        
        # 3. Calculate predicted expectation values
        borns_pred = xp.real(xp.sum(rho * xp.conj(basis), axis=(1, 2)))
        
        # 4. Calculate Loss Gradient w.r.t rho
        diff = borns_pred - borns_approx
        diff_bcast = xp.expand_dims(xp.expand_dims(diff, axis=1), axis=2)
        
        # Normalize the gradient by the number of operators (MSE)
        Delta = (2.0 / num_ops) * xp.sum(diff_bcast * basis, axis=0)
        
        # 5. Analytical Chain Rule to get Gradient w.r.t A
        tr_Delta_A = xp.real(xp.trace(Delta @ A))
        I = xp.eye(dim, dtype=basis.dtype, device=basis.device)
        
        # Use tr_A_safe for gradients to maintain stability
        grad_A = (Delta / tr_A_safe) - (tr_Delta_A / (tr_A_safe**2)) * I
        
        # 6. Analytical Chain Rule to get Gradient w.r.t T
        grad_T = 2.0 * (grad_A @ T)
        
        # 7. Enforce strict lower-triangular structure
        grad_T = xp.tril(grad_T)
        
        # 8. Gradient Clipping to prevent explosion
        max_grad = xp.max(xp.abs(grad_T))
        scale = xp.where(max_grad > clip_limit, clip_limit / max_grad, xp.asarray(1.0, dtype=max_grad.dtype, device=max_grad.device))
        grad_T = grad_T * xp.asarray(scale, dtype=grad_T.dtype, device=grad_T.device)
        
        # 9. Update T
        T = T - lr * grad_T

    # Return the final optimized physical density matrix
    A_final = T @ xp.conj(T).T
    tr_final = xp.real(xp.trace(A_final))
    tr_final_safe = xp.where(tr_final < min_trace, min_trace, tr_final)

    clean_rho = A_final / tr_final_safe
    T_normalized = T / xp.sqrt(tr_final_safe)
    
    return clean_rho, T_normalized

def generate_data(num_particles, num_dims, num_trials, method, xp, device=None):
    total_array = []
    linear_inversion_fids = []

    # --- Precision Fallback for Apple Silicon (MPS) ---
    # MPS lacks native 64-bit float support. We must downcast to 32-bit/64-bit complex.
    is_mps = str(device) == 'mps'
    c_dtype = xp.complex64 if is_mps else xp.complex128
    f_dtype = xp.float32 if is_mps else xp.float64
    
    general_basis_np = pauli_group(num_particles)
    general_basis_xp = xp.asarray(general_basis_np, dtype=c_dtype, device=device)

    reconstruction_basis = general_basis_xp / (num_dims ** num_particles)
    

    for _ in range(num_trials):
        # 1. State Generation (Qiskit runs on CPU, we convert to target hardware)
        if method == "random_mixed":
            matrices_np = random_density_matrix(num_dims**num_particles, rank=None).data
        elif method == "random_pure":
            matrices_np = random_density_matrix(num_dims**num_particles, rank=1).data
        elif method == "haar_random":
            psi_np = random_statevector(num_dims**num_particles).data
            matrices_np = np.outer(psi_np, psi_np.conj())
        elif method == "random_product":
            psi_np = random_statevector(num_dims).data
            for _ in range(num_particles - 1):
                psi_np = np.kron(psi_np, random_statevector(num_dims).data)
            matrices_np = np.outer(psi_np, psi_np.conj())
           
        # Push matrix to target device and apply noise channel
        rho_start0 = xp.asarray(matrices_np, dtype=c_dtype, device=device)
        print("TRACE: ", xp.trace(rho_start0))
        # rho_start0 = global_depo_channel(rho_start0, 0.1, xp)

        # 2. Vectorized Expectation Values
        # Trace(A @ B) is computed efficiently by summing element-wise products 
        borns = xp.asarray([ xp.trace(rho_start0 @ base).real for base in general_basis_xp])

        # borns = np.array([ np.trace(rho_start0 @ base).real for base in general_basis])
        # borns = xp.real(xp.sum(rho_start0 * general_basis_xp, axis=(1, 2)))

        print(f"Borns: {xp.sum(borns)}")

        # 2. Vectorized Expectation Values (Added xp.conj() for Pauli Y correctness)
        # borns = xp.real(xp.sum(rho_start0 * xp.conj(general_basis_xp), axis=(1, 2)))
        
        # 3. Simulate Noise (Generated on CPU, mapped to Device)
        borns = xp.real(xp.sum(rho_start0 * xp.conj(general_basis_xp), axis=(1, 2)))
        print(f"Borns: {xp.sum(borns)}")
        
        # 3. Simulate Noise
        noise_np = norm.rvs(size=num_dims**(2*num_particles)) / 8  
        noise_xp = xp.asarray(noise_np, dtype=f_dtype, device=device)
        borns_approx = xp.asarray(borns + noise_xp, dtype=f_dtype, device=device)
        
        # 4. Vectorized Linear Inversion
        borns_bcast = xp.expand_dims(xp.expand_dims(borns_approx, axis=1), axis=2)
        rback = xp.sum(borns_bcast * reconstruction_basis, axis=0)

        # 5. Eigendecomposition and Matrix Extraction
        cleanRho = eigenvaluesCheck(rback, xp)

        # 4. Maximum Likelihood Estimation (Replaces Linear Inversion)
        # cleanRho, T_opt = mle_reconstruction(borns_approx, general_basis_xp, xp, lr=0.01, iters=300)
        # cleanRho = mle_reconstruction(borns_approx, general_basis_xp, xp, lr=0.1, iters=150)
        
        # 5. Eigendecomposition and Matrix Extraction
        # cleanRho = eigenvaluesCheck(rback, xp)


        try:
            # # Extract Cholesky representation (replaces Linear Inversion)
            # chol_exp = vectorization_new(cleanRho, xp)
            
            # FIX 2: Extract directly from the MLE parameter.
            # chol_exp = extract_cholesky_components(T_opt, xp)

            # 6. Benchmarking metrics
            fid = xp_fidelity(cleanRho, rho_start0, xp)
            linear_inversion_fids.append(float(fid)) 
            
            # 7. Ideal Cholesky target
            chol_theoretic = vectorization_new(PureEigenvaluesCheck(rho_start0, xp), xp)

            # 7. Extract theoretical parameter using safe 64-bit CPU offloading
            # chol_theoretic = get_theoretical_cholesky(rho_start0, xp)

            # Bring tensor back to CPU numpy for safe saving/pandas integration
            combined = xp.concat([chol_exp, chol_theoretic])
            if hasattr(combined, 'cpu'):
                combined_np = combined.cpu().numpy()
            elif hasattr(combined, 'get'):
                combined_np = combined.get() 
            else:
                combined_np = np.array(combined)
                
            total_array.append(combined_np)

        except Exception as e:
            print(f"Trial dropped due to numerical instability: {e}")
            pass

    data = {
        'avg_fid': np.mean(linear_inversion_fids),
        'data_array': np.array(total_array)
    }

    print(f"Mean fidelity ({method}): {data['avg_fid']*100:.4f}%")
    return data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Quantum State Tomography benchmark data.")
    
    parser.add_argument('--num_particles', type=int, default=4, 
                        help='Number of qubits in the system (default: 4)')
    parser.add_argument('--num_dims', type=int, default=2, 
                        help='Dimension of each particle (default: 2 for qubits)')
    parser.add_argument('--num_trials', type=int, default=10, 
                        help='Number of experimental trials to simulate (default: 10)')
    parser.add_argument('--method', type=str, default='random_mixed', 
                        choices=['random_mixed', 'random_pure', 'haar_random', 'random_product'],
                        help='Method to generate quantum states')
    parser.add_argument('--use_cpu', action='store_true', 
                        help='Force the simulation to run on CPU via NumPy')

    # 2. Parse the arguments from the command line
    args = parser.parse_args()

    # 3. Detect hardware dynamically (inverted logic for use_gpu)
    use_gpu = not args.use_cpu
    xp, device = get_compute_backend(use_gpu=use_gpu)
    
    # 4. Pass the parsed arguments to your pipeline
    test_data = generate_data(
        num_particles=args.num_particles, 
        num_dims=args.num_dims, 
        num_trials=args.num_trials, 
        method=args.method, 
        xp=xp, 
        device=device
    )