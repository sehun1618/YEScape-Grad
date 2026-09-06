'''Optional two-channel CUDA ABI for separate Gamma GDF observables.'''

import ctypes
import functools
import os

_GPU4PYSCF_VERSION = '1.8.0'


def check_runtime():
    r'''Require the GPU4PySCF ABI used to build the optional library.

    Raises:
        RuntimeError
            If the installed GPU4PySCF version differs from the audited
            version.  Matching a symbol name alone is not ABI evidence.
    '''
    import gpu4pyscf

    version = str(gpu4pyscf.__version__)
    if version != _GPU4PYSCF_VERSION:
        raise RuntimeError(
            'native_multi_observable_library was audited for GPU4PySCF '
            f'{_GPU4PYSCF_VERSION}, but this process has {version}; rebuild '
            'and revalidate the two-channel ABI first')


@functools.cache
def load_short_range_kernel(library_path):
    r'''Load the audited two-channel GDF short-range CUDA endpoint.

    Args:
        library_path : path-like
            Absolute path of the patched GPU4PySCF ``libpbc`` object.

    Returns:
        ctypes._CFuncPtr
            ``PBCsr_ejk_int3c2e_ip1_two_channel`` with the upstream integer
            status convention.
    '''
    path = os.fspath(library_path)
    check_runtime()
    if not os.path.isabs(path):
        raise ValueError(
            'native_multi_observable_library must be an absolute path')
    if not os.path.isfile(path):
        raise FileNotFoundError(
            'native_multi_observable_library does not exist: ' + path)
    library = ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
    try:
        probe = library.dcforge_pbc_gdf_build_probe
        kernel = library.PBCsr_ejk_int3c2e_ip1_two_channel
    except AttributeError as error:
        raise RuntimeError(
            'native_multi_observable_library lacks the audited DC-Forge '
            'two-channel GDF ABI; build the GPU4PySCF 1.8.0 patch first') from error
    probe.restype = ctypes.c_int
    kernel.restype = ctypes.c_int
    if probe() != 1:
        raise RuntimeError(
            'native_multi_observable_library failed the DC-Forge ABI probe')
    return kernel


def get_short_range_kernel(provider):
    r'''Return the explicit two-channel endpoint configured on ``provider``.

    Args:
        provider : dcforge.gpu.pbc.df.gdf.GDF
            GDF provider whose absolute library path opts into the separate
            target-J and Z-Fock short-range traversal.

    Returns:
        ctypes._CFuncPtr
            Audited two-channel CUDA kernel.
    '''
    library_path = getattr(provider, 'native_multi_observable_library', None)
    if library_path is None:
        raise RuntimeError(
            'separate GPU GDF target-J/Z-Fock ledger requires '
            'with_df.native_multi_observable_library; the default upstream '
            'GPU4PySCF 1.8.0 ABI supports only the fused observable')
    return load_short_range_kernel(library_path)
