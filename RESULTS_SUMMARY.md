# Results Summary

## Main Workflow

The main result is the two-stage seeded ML-kMC workflow:

```text
results/05_seeded_stage1_aggressive/
results/06_seeded_stage2_polish_1000K/
```

The second stage starts from the final structure produced by stage 1, not from the original input structure:

```text
05_seeded_stage1_aggressive -> 06_seeded_stage2_polish_1000K
```

## Comparison Table

| Calculation | Energy start, eV/PuO2 | Energy final, eV/PuO2 | Delta E, eV/PuO2 | Bulk order | Mean coord. error |
| --- | ---: | ---: | ---: | ---: | ---: |
| Baseline | -46.6442 | -46.8323 | -0.1881 | 0.6426 -> 0.6469 | 1.5333 -> 1.5333 |
| Final 5000 steps | -46.6442 | -47.3166 | -0.6724 | 0.6426 -> 0.7914 | 1.5381 -> 1.3000 |
| Seeded + polish | -46.6442 | -47.5877 | -0.9435 | 0.63 -> 0.88 | 1.56 -> 1.12 |

## Interpretation

The seeded two-stage ML-kMC workflow gives the strongest energy decrease and the clearest improvement in local fluorite-like order. The baseline calculation is a control case: it shows that a simple ML-kMC run lowers the energy much less efficiently than the seeded workflow.

The result should be stated carefully. It demonstrates partial defect annealing, energy lowering, and local fluorite-like ordering of PuO2, not complete thermodynamic recrystallization of the whole finite cluster.

## ML Role

The ML model does not replace the physical model. It acts as a fast ranker for candidate events: it predicts energy changes and helps select promising local rearrangements. The selected events are then checked with exact MOX-07 potential energy evaluation.

## Main Final Metrics

For the final structure after seeded two-stage ML-kMC relaxation:

- Energy: `-47.5877 eV/PuO2`
- Bulk order score: `0.8788`
- Mean coordination error: `1.1190`
- Close-contact safety: `True`

## Crystal Visualizations

Initial and final crystal files used for the visualization:

- `input/PuO2_324.xyz`
- `results/06_seeded_stage2_polish_1000K/final.xyz`

Convenience copies:

- `results/07_crystal_visualization/initial_crystal.xyz`
- `results/07_crystal_visualization/final_crystal.xyz`

Generated report figures:

- `results/07_crystal_visualization/initial_crystal_visualization.png`
- `results/07_crystal_visualization/final_crystal_visualization.png`
- `results/07_crystal_visualization/initial_final_comparison.png`
- `results/07_crystal_visualization/final_density_heatmap.png` - 2D density projection integrated over z
- `results/07_crystal_visualization/initial_defect_map.png` - 3D coordination-defect cube map for the initial structure
- `results/07_crystal_visualization/final_defect_map.png` - 3D coordination-defect cube map for the final structure
