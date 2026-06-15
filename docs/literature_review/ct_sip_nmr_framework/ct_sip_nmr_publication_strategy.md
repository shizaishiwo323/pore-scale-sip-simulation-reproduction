# CT-SIP-NMR pore-scale simulation framework: literature review and publication strategy

Date: 2026-06-07

## 1. Search scope

The target manuscript is a methods/framework paper built around:

- micro-CT or segmented digital-rock geometry as pore-scale structural input;
- laboratory SIP or complex conductivity/permittivity spectra as electrical validation data;
- low-field NMR/T2-derived pore-size or fluid-storage constraints as an independent pore-structure validation;
- a self-developed pore-scale AC simulation framework that computes effective complex conductivity/permittivity and compares simulated spectra with experiments.

English search terms used in the first round included:

- `pore-scale simulation spectral induced polarization micro-CT`
- `complex conductivity porous media microtomography`
- `spectral induced polarization porosimetry pore size distribution NMR`
- `nuclear magnetic resonance complex conductivity permeability porous media`
- `digital rock physics effective properties CT segmentation`
- `microfluidics spectral induced polarization calcite dissolution precipitation`

The search used OpenAlex/Crossref/DOI records, publisher pages, and web verification. Two subagents were then used: one audited the relevance of the candidate papers, and one downloaded legally available PDFs or marked unavailable items for manual download.

## 2. Main evidence chain

### 2.1 What the literature clearly proves

The literature clearly proves that there is a mature but still fragmented research space around:

1. micro-CT/digital-rock imaging and pore-scale simulation of effective transport properties;
2. SIP/complex conductivity as a pore-structure-sensitive electrical measurement;
3. NMR as a pore-size and fluid-storage measurement that can be compared with SIP-derived length scales;
4. microfluidic or direct-imaging experiments for interpreting SIP responses during reactive pore-scale processes.

Niu et al. (2020) is the closest direct precedent for a CT-driven pore-scale AC electrical simulation framework. Zhang et al. (2018) and Osterman et al. (2016) support the multi-modal CT/NMR/SIP experimental motivation. Rembert et al. (2023, 2024) and Qiang et al. (2024) show that current high-level work is moving toward direct pore-scale observation plus electrical simulation or petrophysical modeling.

### 2.2 What the literature supports, but does not yet fully solve

The literature supports the idea that SIP relaxation time and quadrature conductivity are linked to pore-size or pore-throat length scales, but the exact physical length scale can be mechanism-dependent. This means the paper should not claim that NMR, CT, and SIP measure the same pore size. A stronger framing is:

> NMR, CT, and SIP provide complementary constraints on pore bodies, pore throats, connectivity, surface area, and electrochemical polarization length scales.

This is precisely where a mechanistic simulation framework can contribute: it can test which pore-scale geometry and polarization assumptions reproduce the measured spectra.

### 2.3 What should be presented as interpretation

The claim that the proposed framework can separate Maxwell-Wagner, pore, membrane, surface/EDL, or grain/interface polarization mechanisms should be phrased as model-based interpretation unless the experiments independently constrain the corresponding surface chemistry and pore-scale charge distributions.

For a strong journal submission, each mechanism should have:

- a clear mathematical source term or constitutive assumption;
- traceable parameters and units;
- a sensitivity analysis;
- an ablation plot showing whether the mechanism is required by the experimental data.

## 3. Core references after audit

### A. Core papers for positioning the novelty

1. Niu, Zhang, and Prasad (2020), *A Framework for Pore-Scale Simulation of Effective Electrical Conductivity and Permittivity of Porous Media in the Frequency Range From 1 mHz to 1 GHz*, Journal of Geophysical Research: Solid Earth. DOI: https://doi.org/10.1029/2020JB020515  
   Use: direct predecessor and benchmark for CT-driven AC electrical pore-scale simulation.

2. Day-Lewis and Johnson (2022), *Pore-Scale Simulation of Spectral Induced Polarization*, OSTI technical report. DOI: https://doi.org/10.2172/1989486  
   Use: pore-network equivalent-circuit route for SIP; useful contrast with voxel/field-solve route.

3. Zhang et al. (2018), *Enhanced pore space analysis by use of μ-CT, MIP, NMR, and SIP*, Solid Earth. DOI: https://doi.org/10.5194/se-9-1225-2018  
   Use: multi-modal pore-space characterization closest to the CT/NMR/SIP experimental package.

4. Osterman et al. (2016), *A laboratory study to estimate pore geometric parameters of sandstones using complex conductivity and nuclear magnetic resonance for permeability prediction*, Water Resources Research. DOI: https://doi.org/10.1002/2015WR018472  
   Use: laboratory NMR plus complex conductivity comparison for pore geometry and permeability.

5. Rembert et al. (2024), *Microfluidics and Spectral Induced Polarization for Direct Observation and Petrophysical Modeling of Calcite Dissolution*, Geophysical Research Letters. DOI: https://doi.org/10.1029/2024GL111271  
   Use: direct pore-scale imaging plus SIP for reactive calcite dissolution.

6. Rembert et al. (2023), *A microfluidic chip for geoelectrical monitoring of critical zone processes*, Lab on a Chip. DOI: https://doi.org/10.1039/D3LC00377A  
   Use: experimental platform for image-SIP coupled microfluidics.

7. Qiang et al. (2024), *Quantitative Evaluation of the Effect of Pore Fluids Distribution on Complex Conductivity Saturation Exponents*, Journal of Geophysical Research: Solid Earth. DOI: https://doi.org/10.1029/2024JB028689  
   Use: recent pore-scale fluid distribution plus complex conductivity simulation.

8. Revil, Florsch, and Camerlynck (2014), *Spectral induced polarization porosimetry*, Geophysical Journal International. DOI: https://doi.org/10.1093/gji/ggu180  
   Use: SIP relaxation/pore-size distribution theory and porosimetry framing.

9. Schwarz (1962), *A theory of the low-frequency dielectric dispersion of colloidal particles in electrolyte solution*, Journal of Physical Chemistry. DOI: https://doi.org/10.1021/j100818a067  
   Use: classical particle/interfacial polarization model background.

### B. Method support

10. Hu and Blunt (2009), *Pore-network extraction from micro-computerized-tomography images*, Physical Review E. DOI: https://doi.org/10.1103/PhysRevE.80.036307  
    Use: maximal-ball pore-network extraction and geometry representation.

11. Raeini, Bijeljic, and Blunt (2017), *Generalized network modeling: Network extraction as a coarse-scale discretization of the void space of porous media*, Physical Review E. DOI: https://doi.org/10.1103/PhysRevE.96.013312  
    Use: network extraction as reduced-order discretization of pore space.

12. Andrae et al. (2012), *Digital rock physics benchmarks - Part I: Imaging and segmentation*, Computers & Geosciences. DOI: https://doi.org/10.1016/j.cageo.2012.09.005  
    Use: CT imaging and segmentation benchmark background.

13. Andrae et al. (2012), *Digital rock physics benchmarks - Part II: Computing effective properties*, Computers & Geosciences. DOI: https://doi.org/10.1016/j.cageo.2012.09.008  
    Use: numerical effective-property computation benchmark background.

14. Blunt et al. (2012), *Pore-scale imaging and modelling*, Advances in Water Resources. DOI: https://doi.org/10.1016/j.advwatres.2012.03.003  
    Use: general digital-rock/pore-scale modeling review.

15. Gostick et al. (2019), *PoreSpy: A Python Toolkit for Quantitative Analysis of Porous Media Images*, Journal of Open Source Software. DOI: https://doi.org/10.21105/joss.01296  
    Use: image analysis and pore-structure metrics.

16. Johnson, Koplik, and Schwartz (1986), *New Pore-Size Parameter Characterizing Transport in Porous Media*, Physical Review Letters. DOI: https://doi.org/10.1103/PhysRevLett.57.2564  
    Use: characteristic pore-size/transport length scale theory.

17. Revil and Florsch (2010), *Determination of permeability from spectral induced polarization in granular media*, Geophysical Journal International. DOI: https://doi.org/10.1111/j.1365-246X.2010.04573.x  
    Use: SIP relaxation and permeability relation.

18. Revil et al. (2015), *Predicting permeability from the characteristic relaxation time and intrinsic formation factor of complex conductivity spectra*, Water Resources Research. DOI: https://doi.org/10.1002/2015WR017074  
    Use: relaxation time, formation factor, and permeability relation.

19. Weller and Slater (2019), *Permeability estimation from induced polarization: an evaluation of geophysical length scales using an effective hydraulic radius concept*, Near Surface Geophysics. DOI: https://doi.org/10.1002/nsg.12071  
    Use: geophysical length scale and hydraulic radius interpretation.

20. Zhang, Niu, and Zhang (2018), *Estimating pore-size distribution in carbonate reservoir rocks using joint inversion of NMR and complex conductivity data*, SEG Technical Program Expanded Abstracts. DOI: https://doi.org/10.1190/segam2018-2997894.1  
    Use: NMR and complex conductivity joint inversion in carbonates.

21. Revil et al. (2017), *Complex conductivity of soils*, Water Resources Research. DOI: https://doi.org/10.1002/2017WR020655  
    Use: surface conduction and complex conductivity model background.

22. Xiong, Baychev, and Jivkov (2016), *Review of pore network modelling of porous media: Experimental characterisations, network constructions and applications to reactive transport*, Journal of Contaminant Hydrology. DOI: https://doi.org/10.1016/j.jconhyd.2016.07.002  
    Use: pore-network modeling background and limitations.

### C. Discussion and background references

23. Kemna et al. (2012), *An overview of the spectral induced polarization method for near-surface applications*, Near Surface Geophysics. DOI: https://doi.org/10.3997/1873-0604.2012027  
    Use: SIP method background.

24. Cnudde and Boone (2013), *High-resolution X-ray computed tomography in geosciences: A review of the current technology and applications*, Earth-Science Reviews. DOI: https://doi.org/10.1016/j.earscirev.2013.04.003  
    Use: CT method background.

25. Anovitz and Cole (2015), *Characterization and Analysis of Porosity and Pore Structures*, Reviews in Mineralogy and Geochemistry. DOI: https://doi.org/10.2138/rmg.2015.80.04  
    Use: pore-structure characterization overview.

26. Jougnot et al. (2009), *Spectral induced polarization of partially saturated clay-rocks: a mechanistic approach*, Geophysical Journal International. DOI: https://doi.org/10.1111/j.1365-246X.2009.04426.x  
    Use: saturation and surface-process mechanism background.

27. Revil and Skold (2011), *Salinity dependence of spectral induced polarization in sands and sandstones*, Geophysical Journal International. DOI: https://doi.org/10.1111/j.1365-246X.2011.05181.x  
    Use: pore-fluid salinity sensitivity.

28. Swanson et al. (2015), *Anomalous solute transport in saturated porous media: Relating transport model parameters to electrical and nuclear magnetic resonance properties*, Water Resources Research. DOI: https://doi.org/10.1002/2014WR015284  
    Use: NMR/electrical connection to transport properties.

29. Niu and Zhang (2019), *Permeability Prediction in Rocks Experiencing Mineral Precipitation and Dissolution: A Numerical Study*, Water Resources Research. DOI: https://doi.org/10.1029/2018WR024174  
    Use: pore-structure evolution and permeability during mineral reactions.

30. Tarasov and Titov (2013), *On the use of the Cole-Cole equations in spectral induced polarization*, Geophysical Journal International. DOI: https://doi.org/10.1093/gji/ggt251  
    Use: caution for Cole-Cole parameter interpretation.

## 4. Download status

Detailed manifest:

- `docs/literature_review/ct_sip_nmr_framework/download_manifest.md`
- `docs/literature_review/ct_sip_nmr_framework/download_manifest.csv`

PDF directory:

- `docs/literature_review/ct_sip_nmr_framework/papers/`

Status summary from the download subagent:

- downloaded from legal open-access sources: 5
- copied from project-local PDFs: 4
- manual download needed: 21

Downloaded or copied PDFs:

- Niu et al. (2020), JGR Solid Earth
- Day-Lewis and Johnson (2022), OSTI report
- Zhang et al. (2018), Solid Earth
- Rembert et al. (2023), Lab on a Chip
- Rembert et al. (2024), GRL
- Hu and Blunt (2009), Physical Review E
- Gostick et al. (2019), JOSS
- Kemna et al. (2012), Near Surface Geophysics
- Cnudde and Boone (2013), Earth-Science Reviews

Manual-download priority:

1. Osterman et al. (2016), WRR, DOI: https://doi.org/10.1002/2015WR018472
2. Qiang et al. (2024), JGR Solid Earth, DOI: https://doi.org/10.1029/2024JB028689
3. Revil et al. (2014), GJI, DOI: https://doi.org/10.1093/gji/ggu180
4. Revil and Florsch (2010), GJI, DOI: https://doi.org/10.1111/j.1365-246X.2010.04573.x
5. Revil et al. (2015), WRR, DOI: https://doi.org/10.1002/2015WR017074
6. Andrae et al. (2012) Part I and Part II, DOI: https://doi.org/10.1016/j.cageo.2012.09.005 and https://doi.org/10.1016/j.cageo.2012.09.008

## 5. What kind of paper can this become?

### Best version: a methods-framework paper

A strong target title would be close to:

> A CT-, SIP-, and NMR-constrained pore-scale simulation framework for effective complex conductivity and permittivity of porous rocks

Core contribution:

- direct use of real 3D CT geometry, not only empirical pore-size distributions;
- multi-modal experimental validation using SIP and NMR from corresponding samples;
- frequency-dependent pore-scale AC field solving;
- mechanism ablation: Maxwell-Wagner, pore, membrane, grain/interface, and combined response;
- reproducible workflow with clear geometry, parameter, solver, and post-processing provenance.

Most suitable target journals if the simulation-experiment agreement is strong:

1. Journal of Geophysical Research: Solid Earth  
   Best if the paper advances rock physics or hydrogeophysics, not only software.

2. Water Resources Research  
   Best if the framework is tied to pore-scale controls on flow, transport, permeability, dissolution/precipitation, or water-rock processes.

3. Geophysical Journal International or Geophysics  
   Best if the paper emphasizes SIP physics, polarization mechanisms, and interpretation of complex conductivity spectra.

4. Computers & Geosciences  
   Best if the main novelty is the computational framework, reproducible code, GPU/FFT solver, data model, visualization, and workflow integration.

5. Transport in Porous Media  
   Best if the core message is porous-media transport physics and upscaling rather than geophysical method development.

### Second version: a digital-rock validation paper

If the experimental match is decent but the new physics is modest, frame it as:

> Digital-rock validation of SIP mechanisms using co-registered CT, NMR, and complex conductivity measurements

Good target journals:

- Computers & Geosciences
- Transport in Porous Media
- Journal of Petroleum Science and Engineering / Geoenergy Science and Engineering
- Petrophysics
- Near Surface Geophysics

### Third version: a reactive microfluidic methods paper

If the strongest result becomes the AC2D/microfluidic calcite dissolution workflow:

> Image-resolved AC simulation of microfluidic SIP responses during calcite dissolution

Good target journals:

- Geophysical Research Letters, if the result is short, mechanism-driven, and surprising.
- Environmental Science & Technology, if the environmental/reactive-process mechanism is central.
- Lab on a Chip, if the contribution includes an experimental platform or strong microfluidic measurement innovation.
- Water Resources Research, if the result advances reactive transport interpretation.

### Lower-risk version: software/data/workflow paper

If the physics comparison is useful but not strong enough for JGR/WRR:

> An open workflow for CT-based pore-scale complex-conductivity simulation and SIP/NMR comparison

Good target journals:

- Computers & Geosciences
- SoftwareX
- Earth Science Informatics
- Journal of Open Source Software, only if the paper is primarily a software note and the code is polished/open.

## 6. Recommended target strategy

The highest-value path is to aim first for JGR: Solid Earth or Water Resources Research, but only if these conditions are met:

1. At least two samples, preferably with contrasting pore structures, have co-located or clearly paired CT, SIP, and NMR data.
2. The framework reproduces not only the order of magnitude but also the frequency trend of real and imaginary conductivity.
3. The NMR/CT pore-size metrics are not merely shown beside SIP; they constrain parameters or explain model mismatch.
4. Mechanism ablation demonstrates why each included polarization mechanism is needed.
5. Uncertainty is explicit: segmentation threshold, voxel size, water conductivity, surface conductivity, relaxation length scale, boundary conditions, and geometry factor.
6. Code/data provenance is strong enough that another researcher can reproduce the main figures.

If those conditions are only partly met, the more realistic first submission is Computers & Geosciences or Transport in Porous Media. That is not a weak outcome: for a self-developed framework, a clean computational-method paper can be more publishable than an overclaimed geophysical-mechanism paper.

## 7. Suggested manuscript structure

1. Introduction  
   Gap: CT/digital-rock, SIP, and NMR are often used separately or linked empirically; few studies close the loop with a pore-scale AC simulation framework validated against paired experiments.

2. Data and samples  
   CT segmentation, SIP setup, NMR processing, sample geometry, saturation and fluid chemistry.

3. Simulation framework  
   Geometry ingestion, phase labeling, complex conductivity assignment, polarization terms, AC field equation, boundary conditions, effective property extraction.

4. Calibration and validation protocol  
   Which parameters are measured, which are fitted, which are fixed from literature, and which are tested by sensitivity analysis.

5. Results  
   CT/NMR pore metrics; simulated versus measured SIP spectra; mechanism ablation; uncertainty envelope.

6. Discussion  
   What length scale SIP appears to sense; where NMR and CT agree/disagree; what mechanisms are necessary; limitations of 2D/3D, segmentation, surface chemistry, and boundary conditions.

7. Conclusions  
   The framework as a bridge from pore-scale structure and multi-modal experiments to effective complex electrical properties.

## 8. One-sentence positioning

If the experiments and simulations match well, this can be positioned as a rock-physics/hydrogeophysics methods-framework paper: a reproducible, CT-resolved pore-scale AC simulation framework constrained and validated by paired SIP and NMR experiments, with mechanism-level interpretation of complex conductivity spectra.

