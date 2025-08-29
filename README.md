A package for fitting previously-correlated FCS data.

# Usage details

For most users, the primary access point should be the notebook `examples/fitting.ipynb` which illustrates the key utilities of the package.
The data is read using the `DelimitedFiles` library whose absolute path should be specified by `filepath`.
Beyond this, the only cell that should be changed from dataset-to-dataset is cell 4 which contains a call to `fcs_plot`.
The desired fitting model is input as the first parameter, followed by the correlation lag times and the correlation value itself at each of the lag times.

Changes should be made to the initial parameter values which are specified by a vector `p`.
The input parameter order depends on which model is selected and the user should refer to the docstrings for detailed references.
Briefly, for 2-dimensional FCS, `fcs_2d`,
*   `p[1]` → τD; the diffusion time
*   `p[2]` → g0; the zero-lag autocorrelation
*   `p[3]` → offset; the offset of the correlation from 0
*   `p[4:m]` → τ_dyn; the dynamic lifetimes
*   `p[m+1:N]` → K_dyn; the fraction corresponding of the population corresponding to the dynamic lifetime.

And for 3-dimensional FCS, `fcs_3d`,
*   `p[1]` → τD; the diffusion time
*   `p[2]` → g0; the zero-lag autocorrelation
*   `p[3]` → offset; the offset of the correlation from 0
*   `p[4]` → s; the structure factor `s = z0/w0`
*   `p[5:m]` → τ_dyn; the dynamic lifetimes
*   `p[m+1:N]` → K_dyn; the fraction corresponding of the population corresponding to the dynamic lifetime.

Each of these parameters are scaled to $\mathcal{O}(1)$ based on their initial guess for the values to stabilize the fitting.

The value of `m` in either case is inferred from the length of the parameter vector.
The "dynamic lifetimes" and their fractions are kept rather general to allow them to encapsulate a number of phenomena including photophysical dark states (e.g., triplets and blinking), PET, and molecule dynamics which are broadly captured by an exponential kernel in the autocorrelation.
To specify which elements of the parameter vector correspond to which physical phenomena, the keyword argument `ics`, short for "independent components" is present.
For instance, if we are to have two triplet states with lifetimes $\tau_1$ and $\tau_2$, we expect that they are dependent on each other in the sense that if a fluorophore is in triplet state 1 it cannot be in triplet state 2.
The result is that the contribution to the autocorrelation is the sum 
$$1 + T_1 e^{- t / \tau_1} + T_2 e^{- t / \tau_2}$$
where $T_1$ and $T_2$ are the fraction of the population in the corresponding triplet state.
On the other hand, if we have one triplet state and one PET location, we expect these events to be independent, amounting to a contribution
$$\left( 1 + T e^{- t / \tau_\mathrm{tr}} \right) \left( 1 + Q e^{- t / \tau_\mathrm{pet}} \right)$$
where $Q$ is the fraction of the observed time spent undergoing PET dynamics.
In the first case, we specify `ics = [2]` since we wish for the two components to be dependent upon each other.
In the second, oen may write `ics = [1,1]`, although this is taken as the base case so such a specification is optional.