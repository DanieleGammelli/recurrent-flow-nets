# Recurrent Flow Networks: Full Repository Coming Soon! (Minimal model code available)

This repository is the official implementation of the RFN, from *Recurrent Flow Networks: A Latent Variable Model for Spatio-Temporal Density Modelling*.

Full paper is available [here](https://arxiv.org/abs/2006.05256)

<table>
  <tr>
    <td>Real Data</td>
     <td>Samples from model</td>
  </tr>
  <tr>
    <td><img align="left" src="images/RFN_data.gif" width="500"/></td>
   <td><img align="left" src="images/RFN_samples.gif" width="500"/></td>
  </tr>
 </table>

<img align="left" src="images/pgm.png" width="1000"/></td>

## Summary

This repository contains:

1. `model.py`: RFN model code
2. `util.py`: utility code
3. `rfn_saved`, `transforms_saved`, `bns_saved`: pre-trained version of the modules characterizing the RFN
4. `/data`: folder containing data used for the NYC-P experiment

## Training and Evaluation code

A working Jupyter Notebook is provided in `rfn_nyp.ipynb`, showing a basic usage of the proposed RFN for the NYC-P task (more details in Section 3 of the paper).

The notebook contains:

1. Loading and processing of data
2. Building RFN object
3. Train or loading pre-trained model code
4. Evaluation code
5. Basic visualizations
