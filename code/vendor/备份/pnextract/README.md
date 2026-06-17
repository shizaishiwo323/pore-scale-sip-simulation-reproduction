



![make and test](https://github.com/aliraeini/pnextract/workflows/make%20and%20test/badge.svg)

##  pnextract - pore-network extraction
**pnextract** is an open-source software written in C++ that extracts pore networks from images or statistical reconstructions of porous materials. These pore networks preserve the topology of the pore space and also calculate other parameters, such as the pore size distribution, interfacial area, and volume needed for analysis and simulations. Network extraction is normally the first step for further analysis and modelling, to predict flow and transport processes in porous materials, and to analyse three-dimensional images. This code provides essential network files for pore network flow model ([**pnflow**](https://github.com/aliraeini/pnflow)). 

**Note: this repository is same as [pnflow repository](https://github.com/aliraeini/pnflow) but without the pnflow code.**


 ----------------------------------------------------------------

## See [src/pnm](src/pnm) and [doc](doc) for details on pnextract code.

## See [src/script/README.md](src/script/README.md) for compile/build instructions.

## Local Niu 2020 Berea contact-split V2 defaults

This project-local vendor copy includes the contact-split conductance-conserving
`writePNM` export in `src/pnm/pnextract/blockNet_write_cnm.cpp` and a calibrated
Windows executable at `bin/pnextract_contact_split_v2.exe`.

For the Niu 2020 Berea reproduction, append the parameter lines stored in
`config/niu2020_contact_split_conserve_pnextract_lines.txt` to the generated
`.mhd` input before running pnextract:

```text
minRPore 0.02
medialSurfaceSettings 0.02 0.98 0.85 0.1 1.8 2.2 0 0 0.05
```

This setting reproduces the calibrated contact-split V2 geometry:
15439 pores, 19743 split throats, and throat L1 distance about 0.364497 against
Niu Figure5. The outer repository copy at `C:\Users\imgw\Documents\Codex\SIP模拟\pnextract`
remains the original backup.

See also README files of other modules which are located in their own directories:    
[src/libvoxel](src/libvoxel), [src/script](src/script) and in [thirdparty](thirdparty).


 ----------------------------------------------------------------

Download the [bin.7z](bin.7z) for pre-compiled Windows executables. 

### Contact and References ###

For contacts and references please see: 
https://www.imperial.ac.uk/earth-science/research/research-groups/pore-scale-modelling


Alternatively, contact Sajjad Foroughi:
- Email: s.foroughi@imperial.ac.uk
- Additional Email: foroughi.sajad@gmail.com

--

For more in-depth understanding of the pnextract code, users can refer to following papers related to this topic, which may provide additional insights:
- [Pore-network extraction from micro-computerized-tomography images](https://journals.aps.org/pre/abstract/10.1103/PhysRevE.80.036307)
- [Generalized network modeling: Network extraction as a coarse-scale discretization of the void space of porous media](https://journals.aps.org/pre/abstract/10.1103/PhysRevE.96.013312) 
