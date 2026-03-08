"""
CKKS parameter selection and context management utilities.

Wraps TenSEAL for convenient experiment setup.
"""

import tenseal as ts
from typing import List, Optional


def make_ckks_context(
    depth: int,
    scale_bits: int = 40,
    security_level: int = 128,
) -> ts.Context:
    """
    Create a TenSEAL CKKS context for a given multiplicative depth.

    Automatically selects polynomial degree based on depth and security.

    Args:
        depth: Required multiplicative depth (number of levels)
        scale_bits: Bits of precision for scale (default 40 = ~12 decimal digits)
        security_level: Target security in bits (128 or 192)

    Returns:
        Configured TenSEAL context
    """
    # Build modulus chain: [special] + [depth levels] + [special]
    # Each level: scale_bits; first and last: 60 bits
    coeff_mod_bit_sizes = [60] + [scale_bits] * depth + [60]
    total_bits = sum(coeff_mod_bit_sizes)

    # Choose n based on total modulus bits and security
    # Reference: HomomorphicEncryption.org standard Table 2
    if security_level == 128:
        if total_bits <= 218:
            poly_degree = 8192
        elif total_bits <= 438:
            poly_degree = 16384
        elif total_bits <= 881:
            poly_degree = 32768
        else:
            poly_degree = 65536
    elif security_level == 192:
        if total_bits <= 161:
            poly_degree = 8192
        elif total_bits <= 323:
            poly_degree = 16384
        elif total_bits <= 669:
            poly_degree = 32768
        else:
            poly_degree = 65536
    else:
        raise ValueError(f"Unsupported security level: {security_level}")

    context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=poly_degree,
        coeff_mod_bit_sizes=coeff_mod_bit_sizes,
    )
    context.global_scale = 2**scale_bits
    context.generate_galois_keys()  # needed for rotations (attention)

    return context


def context_info(ctx: ts.Context) -> dict:
    """Return a summary of context parameters."""
    return {
        "poly_modulus_degree": ctx.poly_modulus_degree(),
        "global_scale_bits": ctx.global_scale.bit_length() if hasattr(ctx.global_scale, 'bit_length') else None,
        "n_slots": ctx.poly_modulus_degree() // 2,
    }


def encrypt_vector(ctx: ts.Context, values: List[float]) -> ts.CKKSVector:
    """Encrypt a list of floats as a CKKS vector."""
    return ts.ckks_vector(ctx, values)


def decrypt_vector(enc_vec: ts.CKKSVector) -> List[float]:
    """Decrypt a CKKS vector."""
    return enc_vec.decrypt()
