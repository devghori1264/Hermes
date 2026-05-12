Hermes External Reproducibility Package
=======================================

1. Purpose
This document provides the exact replication checklist for independent review teams. Our goal is 100 percent transparency and reproducibility.

2. Replication Checklist
* Verify the dataset checksums against the signed manifests.
* Execute the offline benchmark matrix using the provided seeds.
* Validate the entity resolution output against the baseline labels.
* Rerun the causal estimators to confirm the average treatment effects.
* Inspect the leakage reports to ensure zero target correlation.
* Confirm that the fairness aware reranker blocks overexposed cohorts.

3. Manifest Usage
All experiments are logged in the registry using the ExperimentManifest structure. Reviewers must load the manifest by ID to restore the exact parameters and model versions used during training.

4. Verification
Once all checklist items pass, the external reviewer may sign off on the elite maturity gate.
