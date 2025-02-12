import numpy as np

# -------------------------
# Normalize Feature Values Using Percentiles
# -------------------------
# Define a function that uses the 5th and 95th percentiles for normalization
def normalize_feature(x):
    vmin = np.nanpercentile(x, 5)
    vmax = np.nanpercentile(x, 95)
    if vmin == vmax:  # fallback if there is no variation
        vmin = x.min()
        vmax = x.max()
    norm = (x - vmin) / (vmax - vmin)
    norm = np.clip(norm, 0, 1)
    return norm


# -------------------------
# Define a function to compute jitter based on SHAP value distribution per feature,
# similar to the logic in the SHAP beeswarm plot.
# -------------------------
def compute_beeswarm_jitter(shap_vals, row_height=0.4, nbins=100):
    """
    Compute a jitter offset for an array of SHAP values in the style of the SHAP beeswarm plot.
    
    Parameters:
        shap_vals (np.array): 1D array of SHAP values for one feature.
        row_height (float): The height allocated for one row in the beeswarm plot.
        nbins (int): Number of bins to group the SHAP values.
    
    Returns:
        np.array: Jitter offsets (of the same shape as shap_vals).
    """
    N = len(shap_vals)
    # Add a tiny noise term for stable sorting
    noise = np.random.randn(N) * 1e-6
    min_val = np.min(shap_vals)
    max_val = np.max(shap_vals)
    # Bin the SHAP values into one of 'nbins' bins
    quant = np.round(nbins * (shap_vals - min_val) / (max_val - min_val + 1e-8))
    # Get the order of indices based on the binned value plus noise
    inds_sorted = np.argsort(quant + noise)
    
    jitter = np.zeros(N)
    layer = 0
    last_bin = -1
    # Iterate over the sorted indices, assigning offsets based on how many points are in the same bin
    for idx in inds_sorted:
        if quant[idx] != last_bin:
            layer = 0
        # Alternate the sign of the offset and increase the magnitude gradually
        jitter[idx] = np.ceil(layer / 2) * ((layer % 2) * 2 - 1)
        layer += 1
        last_bin = quant[idx]
    
    # Scale the jitter to be within the row height (using 0.9 as a scaling factor)
    max_abs = np.max(np.abs(jitter)) if np.max(np.abs(jitter)) != 0 else 1
    jitter = jitter * (0.9 * (row_height / max_abs))
    return jitter
